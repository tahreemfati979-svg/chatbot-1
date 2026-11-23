export async function GET(): Promise<Response> {
  return new Response(
    JSON.stringify({
      ok: true,
      model: 'gemini-3.1-flash-lite',
      hasApiKey: Boolean(
        process.env.GEMINI_API_KEY?.trim() &&
          process.env.GEMINI_API_KEY.trim() !== 'MY_GEMINI_API_KEY'
      ),
    }),
    {
      status: 200,
      headers: { 'Content-Type': 'application/json; charset=utf-8' },
    }
  );
}
