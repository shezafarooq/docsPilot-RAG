import { forwardHistory } from '../../../../lib/proxy.mjs';

export const runtime = 'nodejs';

export async function GET(request) {
  return forwardHistory(request, { baseUrl: process.env.API_BASE_URL });
}
