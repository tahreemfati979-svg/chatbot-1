import { GoogleGenAI, ThinkingLevel } from '@google/genai';

const PRIMARY_MODEL = 'gemini-3.1-flash-lite';
const FALLBACK_MODEL = 'gemini-4-hydrogen';
const SYSTEM_INSTRUCTION =
  'You are Lumina, a fast, helpful, and accurate AI assistant. Give clear, concise answers and use clean Markdown when helpful. If asked for live real-time information (such as current weather, live scores, or breaking news) that you cannot verify, clearly state that you do not have real-time internet access rather than guessing.';

interface IncomingMessage {
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

function validateMessages(raw: unknown): IncomingMessage[] | null {
  if (!Array.isArray(raw) || raw.length === 0) {
    return null;
  }

  const sanitized: IncomingMessage[] = [];
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

function jsonResponse(body: Record<string, unknown>, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json; charset=utf-8' },
  });
}

export async function POST(request: Request): Promise<Response> {
  const ai = getAiClient();
  if (!ai) {
    return jsonResponse(
      {
        code: 'MISSING_API_KEY',
        error:
          'GEMINI_API_KEY is not configured. Add GEMINI_API_KEY in your environment variables (Vercel Project Settings > Environment Variables or Settings > Secrets).',
      },
      503
    );
  }

  let body: { messages?: unknown; stream?: boolean } | null = null;
  try {
    body = (await request.json()) as { messages?: unknown; stream?: boolean };
  } catch {
    return jsonResponse({ code: 'INVALID_JSON', error: 'Invalid JSON request body.' }, 400);
  }

  const messages = validateMessages(body?.messages);
  if (!messages) {
    return jsonResponse(
      {
        code: 'INVALID_REQUEST',
        error: 'Please provide a valid non-empty message history ending with a user message.',
      },
      400
    );
  }

  const useStream = body?.stream !== false;
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
      return jsonResponse({ text: response.text ?? '' }, 200);
    } catch (err) {
      const parsed = parseApiError(err);
      return jsonResponse({ code: parsed.code, error: parsed.message }, parsed.status);
    }
  }

  try {
    let geminiStream;
    try {
      geminiStream = await withRetry(() =>
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
      geminiStream = await withRetry(() =>
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

    const encoder = new TextEncoder();
    const readable = new ReadableStream({
      async start(controller) {
        try {
          for await (const chunk of geminiStream) {
            if (request.signal.aborted) {
              break;
            }
            const text = chunk.text;
            if (text) {
              controller.enqueue(encoder.encode(`data: ${JSON.stringify({ text })}\n\n`));
            }
          }
          if (!request.signal.aborted) {
            controller.enqueue(encoder.encode('data: [DONE]\n\n'));
          }
        } catch (err) {
          const parsed = parseApiError(err);
          if (!request.signal.aborted) {
            controller.enqueue(
              encoder.encode(`data: ${JSON.stringify({ error: parsed.message, code: parsed.code })}\n\n`)
            );
          }
        } finally {
          controller.close();
        }
      },
    });

    return new Response(readable, {
      status: 200,
      headers: {
        'Content-Type': 'text/event-stream; charset=utf-8',
        'Cache-Control': 'no-cache, no-transform',
        Connection: 'keep-alive',
        'X-Accel-Buffering': 'no',
      },
    });
  } catch (err) {
    const parsed = parseApiError(err);
    return jsonResponse({ code: parsed.code, error: parsed.message }, parsed.status);
  }
}
