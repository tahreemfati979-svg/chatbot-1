import type { IncomingMessage, ServerResponse } from 'node:http';
import { GoogleGenAI, ThinkingLevel } from '@google/genai';

const PRIMARY_MODEL = 'gemini-3.1-flash-lite';
const FALLBACK_MODEL = 'gemini-4-hydrogen';
const SYSTEM_INSTRUCTION =
  'You are Lumina, a fast, helpful, and accurate AI assistant. Give clear, concise answers and use clean Markdown when helpful. If asked for live real-time information (such as current weather, live scores, or breaking news) that you cannot verify, clearly state that you do not have real-time internet access rather than guessing.';

interface VercelRequest extends IncomingMessage {
  body?: unknown;
}

interface IncomingChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

function getAiClient(): GoogleGenAI | null {
  const apiKey = process.env.GEMINI_API_KEY?.trim();
  if (!apiKey || apiKey === 'MY_GEMINI_API_KEY') {
    return null;
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

function sendJson(res: ServerResponse, status: number, payload: Record<string, unknown>): void {
  res.statusCode = status;
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.end(JSON.stringify(payload));
}

async function readJsonBody(req: VercelRequest): Promise<Record<string, unknown> | null> {
  if (req.body && typeof req.body === 'object') {
    return req.body as Record<string, unknown>;
  }
  if (typeof req.body === 'string') {
    try {
      return JSON.parse(req.body) as Record<string, unknown>;
    } catch {
      return null;
    }
  }

  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(typeof chunk === 'string' ? Buffer.from(chunk) : chunk);
  }
  if (chunks.length === 0) return null;
  try {
    return JSON.parse(Buffer.concat(chunks).toString('utf8')) as Record<string, unknown>;
  } catch {
    return null;
  }
}

function parseApiError(err: unknown): { status: number; message: string; code: string } {
  const errObj = err as { status?: number; message?: string };
  const statusCode = typeof errObj?.status === 'number' ? errObj.status : 0;
  const rawMessage = err instanceof Error ? err.message : String(err);
  const lower = rawMessage.toLowerCase();

  console.error('[Gemini API Error]', { statusCode, message: rawMessage.slice(0, 300) });

  if (
    statusCode === 429 ||
    lower.includes('429') ||
    lower.includes('resource_exhausted') ||
    lower.includes('quota') ||
    lower.includes('rate')
  ) {
    return {
      status: 429,
      code: 'RATE_LIMIT',
      message: 'Rate limit reached on the free Gemini tier. Please wait a moment and try again.',
    };
  }

  if (
    statusCode === 401 ||
    statusCode === 403 ||
    lower.includes('api_key_invalid') ||
    lower.includes('api key not valid') ||
    lower.includes('permission_denied') ||
    lower.includes('401') ||
    lower.includes('403')
  ) {
    return {
      status: 403,
      code: 'AUTH_ERROR',
      message: 'Gemini API authentication failed. Please verify your GEMINI_API_KEY environment variable.',
    };
  }

  if (statusCode === 404 || lower.includes('not_found') || lower.includes('404')) {
    return {
      status: 404,
      code: 'MODEL_NOT_FOUND',
      message: 'The configured AI model is currently unavailable. Please try again shortly.',
    };
  }

  if (statusCode === 400 || lower.includes('invalid_argument')) {
    return {
      status: 400,
      code: 'BAD_REQUEST',
      message: 'The request could not be processed by Gemini. Please check your input or API key.',
    };
  }

  if (
    lower.includes('network') ||
    lower.includes('fetch') ||
    lower.includes('econnreset') ||
    lower.includes('etimedout')
  ) {
    return {
      status: 503,
      code: 'NETWORK_ERROR',
      message: 'Unable to reach the Gemini service due to a temporary network issue. Please retry.',
    };
  }

  return {
    status: 500,
    code: 'UPSTREAM_ERROR',
    message: 'Something went wrong while generating a response. Please try again.',
  };
}

async function withRetry<T>(fn: () => Promise<T>, maxRetries = 2): Promise<T> {
  let lastError: unknown;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;
      const msg = err instanceof Error ? err.message.toLowerCase() : '';
      const isTransient =
        msg.includes('503') ||
        msg.includes('unavailable') ||
        msg.includes('etimedout') ||
        msg.includes('econnreset');

      if (!isTransient || attempt === maxRetries) {
        throw err;
      }

      const delayMs = 400 * Math.pow(2, attempt);
      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
  }
  throw lastError;
}

function validateMessages(raw: unknown): IncomingChatMessage[] | null {
  if (!Array.isArray(raw) || raw.length === 0) {
    return null;
  }

  const sanitized: IncomingChatMessage[] = [];
  for (const item of raw.slice(-20)) {
    if (!item || typeof item !== 'object') return null;
    const role = (item as Record<string, unknown>).role;
    const content = (item as Record<string, unknown>).content;
    if ((role !== 'user' && role !== 'assistant') || typeof content !== 'string') {
      return null;
    }
    const trimmed = content.trim();
    if (!trimmed) continue;
    sanitized.push({
      role,
      content: trimmed.slice(0, 32000),
    });
  }

  if (sanitized.length === 0 || sanitized[sanitized.length - 1].role !== 'user') {
    return null;
  }

  return sanitized;
}

export default async function handler(req: VercelRequest, res: ServerResponse): Promise<void> {
  if (req.method === 'OPTIONS') {
    res.statusCode = 204;
    res.end();
    return;
  }

  if (req.method !== 'POST') {
    sendJson(res, 405, { code: 'METHOD_NOT_ALLOWED', error: 'Method not allowed. Use POST.' });
    return;
  }

  const ai = getAiClient();
  if (!ai) {
    sendJson(res, 503, {
      code: 'MISSING_API_KEY',
      error:
        'GEMINI_API_KEY is not configured. Add GEMINI_API_KEY in Vercel Project Settings > Environment Variables and redeploy.',
    });
    return;
  }

  const body = await readJsonBody(req);
  if (!body) {
    sendJson(res, 400, { code: 'INVALID_JSON', error: 'Invalid JSON request body.' });
    return;
  }

  const messages = validateMessages(body.messages);
  if (!messages) {
    sendJson(res, 400, {
      code: 'INVALID_REQUEST',
      error: 'Please provide a valid non-empty message history ending with a user message.',
    });
    return;
  }

  const useStream = body.stream !== false;
  const contents = messages.map((msg) => ({
    role: msg.role === 'assistant' ? 'model' : 'user',
    parts: [{ text: msg.content }],
  }));

  if (!useStream) {
    try {
      let response;
      try {
        response = await withRetry(() =>
          ai.models.generateContent({
            model: PRIMARY_MODEL,
            contents,
            config: {
              systemInstruction: SYSTEM_INSTRUCTION,
              thinkingConfig: { thinkingLevel: ThinkingLevel.MINIMAL },
            },
          })
        );
      } catch (primaryErr) {
        const parsedPrimary = parseApiError(primaryErr);
        if (parsedPrimary.status !== 404) throw primaryErr;
        response = await withRetry(() =>
          ai.models.generateContent({
            model: FALLBACK_MODEL,
            contents,
            config: {
              systemInstruction: SYSTEM_INSTRUCTION,
              thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
            },
          })
        );
      }
      sendJson(res, 200, { text: response.text ?? '' });
    } catch (err) {
      const parsed = parseApiError(err);
      sendJson(res, parsed.status, { code: parsed.code, error: parsed.message });
    }
    return;
  }

  let headersSent = false;
  let clientClosed = false;
  res.on('close', () => {
    if (!res.writableEnded) {
      clientClosed = true;
    }
  });

  try {
    let stream;
    try {
      stream = await withRetry(() =>
        ai.models.generateContentStream({
          model: PRIMARY_MODEL,
          contents,
          config: {
            systemInstruction: SYSTEM_INSTRUCTION,
            thinkingConfig: { thinkingLevel: ThinkingLevel.MINIMAL },
          },
        })
      );
    } catch (primaryErr) {
      const parsedPrimary = parseApiError(primaryErr);
      if (parsedPrimary.status !== 404) throw primaryErr;
      stream = await withRetry(() =>
        ai.models.generateContentStream({
          model: FALLBACK_MODEL,
          contents,
          config: {
            systemInstruction: SYSTEM_INSTRUCTION,
            thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          },
        })
      );
    }

    res.statusCode = 200;
    res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
    res.setHeader('Cache-Control', 'no-cache, no-transform');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no');
    res.flushHeaders?.();
    headersSent = true;

    for await (const chunk of stream) {
      if (clientClosed) break;
      const text = chunk.text;
      if (text) {
        res.write(`data: ${JSON.stringify({ text })}\n\n`);
      }
    }

    if (!clientClosed) {
      res.write('data: [DONE]\n\n');
      res.end();
    }
  } catch (err) {
    const parsed = parseApiError(err);
    if (!headersSent) {
      sendJson(res, parsed.status, { code: parsed.code, error: parsed.message });
    } else if (!clientClosed) {
      res.write(`data: ${JSON.stringify({ error: parsed.message, code: parsed.code })}\n\n`);
      res.end();
    }
  }
}
