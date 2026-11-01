import { forwardQuestion } from '../../../lib/proxy.mjs';

export const runtime = 'nodejs';

export async function POST(request) {
  return forwardQuestion(request, { baseUrl: process.env.API_BASE_URL });
}
