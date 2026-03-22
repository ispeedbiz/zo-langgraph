-- Seed 6 initial Builder Capability Modules (BCMs)
-- These are knowledge packages the Builder sub-minds read before coding
-- Each covers a specialized domain the standard 5 sub-minds don't know

BEGIN;

-- 1. PDF Generation
INSERT INTO zo_builder_modules (module_id, name, description, capabilities, content, quality_score)
VALUES (
  'bcm-pdf-generation',
  'PDF & Document Generation',
  'Generate professional PDFs (invoices, reports, letters) in Next.js',
  ARRAY['pdf-generation', 'invoice-pdf', 'report-pdf', 'document-template'],
  '# BCM: PDF & Document Generation

## What This Module Provides
Technical knowledge for generating professional PDF documents in a Next.js/React application.

## Recommended Libraries
### @react-pdf/renderer (Primary — client-side)
- Pure React components that render to PDF
- No server dependency, works in browser
- Install: `npm install @react-pdf/renderer`
- Best for: invoices, letters, simple reports

### jsPDF + html2canvas (Fallback — client-side)
- Convert HTML/CSS to PDF
- More flexible styling but lower quality
- Install: `npm install jspdf html2canvas`

### Puppeteer/Playwright (Server-side — Supabase Edge Function)
- Headless Chrome renders HTML to PDF
- Highest quality, supports full CSS
- Heavy — use only when client-side insufficient

## Code Patterns

### React-PDF Invoice Component
```tsx
import { Document, Page, Text, View, StyleSheet } from "@react-pdf/renderer";

const Invoice = ({ data }) => (
  <Document>
    <Page size="A4" style={styles.page}>
      <View style={styles.header}>
        <Text style={styles.title}>INVOICE #{data.number}</Text>
        <Text>{data.date}</Text>
      </View>
      <View style={styles.table}>
        {data.items.map((item, i) => (
          <View key={i} style={styles.row}>
            <Text style={styles.cell}>{item.description}</Text>
            <Text style={styles.amount}>${item.amount.toFixed(2)}</Text>
          </View>
        ))}
      </View>
      <View style={styles.total}>
        <Text>Total: ${data.total.toFixed(2)}</Text>
      </View>
    </Page>
  </Document>
);
```

### Download Button
```tsx
import { PDFDownloadLink } from "@react-pdf/renderer";

<PDFDownloadLink document={<Invoice data={invoiceData} />} fileName="invoice.pdf">
  {({ loading }) => loading ? "Generating..." : "Download PDF"}
</PDFDownloadLink>
```

### Server-side via API Route
```ts
// app/api/pdf/route.ts
import { renderToBuffer } from "@react-pdf/renderer";

export async function POST(req: Request) {
  const data = await req.json();
  const buffer = await renderToBuffer(<Invoice data={data} />);
  return new Response(buffer, {
    headers: { "Content-Type": "application/pdf", "Content-Disposition": "attachment; filename=invoice.pdf" }
  });
}
```

## Integration Points
- Supabase Storage: save generated PDFs for later access
- Stripe: attach PDF receipt to payment confirmation email
- Email: send PDF as attachment via Resend API

## Known Gotchas
1. @react-pdf/renderer does NOT support all CSS — no flexbox gap, limited grid
2. Fonts must be registered explicitly (no system fonts in PDF)
3. Images must be absolute URLs or base64 (no relative paths)
4. Server-side rendering requires Node.js runtime (not Edge)
5. Large PDFs (50+ pages) may timeout on serverless — stream instead

## Testing Approach
- Unit test: verify PDF component renders without errors
- Snapshot test: compare generated PDF buffer size
- Visual test: generate PDF, convert to image, compare',
  7.0
)
ON CONFLICT (module_id) DO NOTHING;

-- 2. Voice & Speech Processing
INSERT INTO zo_builder_modules (module_id, name, description, capabilities, content, quality_score)
VALUES (
  'bcm-voice-speech',
  'Voice & Speech Processing',
  'Voice input capture, speech-to-text, and audio processing in the browser',
  ARRAY['voice-to-text', 'speech-recognition', 'audio-capture', 'voice-input'],
  '# BCM: Voice & Speech Processing

## What This Module Provides
Technical knowledge for implementing voice input and speech-to-text in Next.js/React.

## API Reference

### Web Speech API (SpeechRecognition) — Primary
- Browser support: Chrome, Edge, Safari 14.1+
- No API key required — runs entirely in browser
- Free, unlimited usage
- Key classes: SpeechRecognition, SpeechRecognitionEvent

### Whisper API (OpenAI) — Fallback for unsupported browsers
- Server-side transcription
- Cost: $0.006/minute
- Supports 50+ languages

## Code Patterns

### React Hook: useVoiceInput
```tsx
import { useState, useCallback, useRef } from "react";

export function useVoiceInput() {
  const [transcript, setTranscript] = useState("");
  const [isListening, setIsListening] = useState(false);
  const recognitionRef = useRef<SpeechRecognition | null>(null);

  const startListening = useCallback(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert("Voice input not supported in this browser");
      return;
    }
    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onresult = (event) => {
      let finalTranscript = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript;
        }
      }
      if (finalTranscript) setTranscript(prev => prev + " " + finalTranscript);
    };

    recognition.onerror = (event) => {
      console.error("Speech error:", event.error);
      setIsListening(false);
    };

    recognition.onend = () => setIsListening(false);
    recognitionRef.current = recognition;
    recognition.start();
    setIsListening(true);
  }, []);

  const stopListening = useCallback(() => {
    recognitionRef.current?.stop();
    setIsListening(false);
  }, []);

  return { transcript, isListening, startListening, stopListening, setTranscript };
}
```

## Integration Points
- Claude API: parse transcript into structured data (invoice fields, etc.)
- Supabase: store transcripts temporarily, delete after processing
- Next.js API route: /api/voice/parse for server-side fallback

## Known Gotchas
1. Safari requires webkit prefix: window.webkitSpeechRecognition
2. HTTPS required — will not work on HTTP (except localhost)
3. Mobile Chrome requires user gesture (button tap) to start
4. Background noise: implement confidence threshold (result[0].confidence > 0.7)
5. Privacy: always show clear recording indicator, never record without consent

## Testing Approach
- Mock SpeechRecognition in Jest/Vitest tests
- E2E: Playwright cannot access microphone — test with mocked audio
- Manual QA: test on Chrome, Safari, mobile Chrome',
  6.5
)
ON CONFLICT (module_id) DO NOTHING;

-- 3. Email Integration
INSERT INTO zo_builder_modules (module_id, name, description, capabilities, content, quality_score)
VALUES (
  'bcm-email-integration',
  'Email Integration & Automation',
  'Send transactional emails, welcome sequences, and notifications via Resend API',
  ARRAY['email-send', 'email-template', 'email-sequence', 'transactional-email'],
  '# BCM: Email Integration

## What This Module Provides
Technical knowledge for sending transactional emails in Next.js using Resend API.

## API Reference

### Resend API (Primary — ZeroOrigine standard)
- Base URL: https://api.resend.com
- Auth: Bearer token in Authorization header
- Free tier: 100 emails/day, 3000/month
- React Email for templates

## Code Patterns

### Send Email via API Route
```ts
// app/api/email/send/route.ts
export async function POST(req: Request) {
  const { to, subject, html } = await req.json();
  const res = await fetch("https://api.resend.com/emails", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${process.env.RESEND_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      from: "ProductName <noreply@zeroorigine.com>",
      to, subject, html,
    }),
  });
  return Response.json(await res.json());
}
```

### React Email Template
```tsx
import { Html, Head, Body, Container, Text, Button } from "@react-email/components";

export const WelcomeEmail = ({ name, productName }) => (
  <Html>
    <Head />
    <Body style={{ fontFamily: "sans-serif", background: "#f4f4f5" }}>
      <Container style={{ maxWidth: 600, margin: "0 auto", padding: 20 }}>
        <Text style={{ fontSize: 24 }}>Welcome to {productName}!</Text>
        <Text>Hi {name}, your account is ready.</Text>
        <Button href="https://app.example.com" style={{ background: "#000", color: "#fff", padding: "12px 24px" }}>
          Get Started
        </Button>
      </Container>
    </Body>
  </Html>
);
```

## Integration Points
- Supabase Auth webhook: trigger welcome email on signup
- Stripe webhook: send receipt email on payment
- Cron/n8n: scheduled digest emails

## Known Gotchas
1. Domain verification required for custom from address
2. Rate limits: 100/day on free tier — queue if needed
3. HTML email rendering varies wildly across clients — test with Litmus/Email on Acid
4. Always include unsubscribe link (CAN-SPAM compliance)
5. SPF/DKIM records must be set on zeroorigine.com DNS',
  7.0
)
ON CONFLICT (module_id) DO NOTHING;

-- 4. Data Pipeline & External APIs
INSERT INTO zo_builder_modules (module_id, name, description, capabilities, content, quality_score)
VALUES (
  'bcm-data-pipeline',
  'Data Pipeline & External API Aggregation',
  'Fetch, normalize, and cache data from external APIs and web sources',
  ARRAY['data-pipeline', 'api-aggregation', 'web-scraping', 'data-normalization'],
  '# BCM: Data Pipeline & External APIs

## What This Module Provides
Patterns for fetching, normalizing, and caching data from external sources.

## Architecture Patterns

### Supabase Edge Function (Recommended)
- Runs on Deno, close to database
- 50ms cold start, 150MB memory
- Perfect for API aggregation

### Next.js API Route + Cron
- Use for scheduled data fetches
- Vercel Cron or n8n trigger

## Code Patterns

### Edge Function: Fetch & Cache
```ts
// supabase/functions/fetch-grants/index.ts
import { createClient } from "@supabase/supabase-js";

Deno.serve(async (req) => {
  const supabase = createClient(
    Deno.env.get("SUPABASE_URL")!,
    Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!
  );

  // Fetch from external API
  const grants = await fetch("https://api.grants.gov/v2/opportunities").then(r => r.json());

  // Normalize
  const normalized = grants.map(g => ({
    external_id: g.id,
    title: g.title,
    amount_min: g.award?.floor || 0,
    amount_max: g.award?.ceiling || 0,
    deadline: g.close_date,
    category: g.category?.name || "general",
    source: "grants.gov",
    fetched_at: new Date().toISOString(),
  }));

  // Upsert (idempotent)
  const { error } = await supabase.from("external_data")
    .upsert(normalized, { onConflict: "external_id" });

  return new Response(JSON.stringify({ count: normalized.length, error }));
});
```

### Rate Limiting & Retry
```ts
async function fetchWithRetry(url: string, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const res = await fetch(url);
      if (res.status === 429) {
        await new Promise(r => setTimeout(r, 2 ** i * 1000));
        continue;
      }
      return await res.json();
    } catch (e) {
      if (i === maxRetries - 1) throw e;
    }
  }
}
```

## Known Gotchas
1. Always cache external data — never query live on user request
2. Handle API key rotation (store in Supabase Vault or env vars)
3. Respect rate limits — implement exponential backoff
4. Data freshness: set cache TTL per source (grants: 24h, prices: 1h)
5. Legal: check API terms for data storage and display rights',
  6.0
)
ON CONFLICT (module_id) DO NOTHING;

-- 5. Calendar & Scheduling
INSERT INTO zo_builder_modules (module_id, name, description, capabilities, content, quality_score)
VALUES (
  'bcm-calendar-api',
  'Calendar & Scheduling Integration',
  'Google Calendar, Outlook Calendar, and time-based scheduling',
  ARRAY['calendar-api', 'google-calendar', 'scheduling', 'time-slots'],
  '# BCM: Calendar & Scheduling

## What This Module Provides
Integration patterns for calendar APIs and scheduling features.

## API Reference

### Google Calendar API
- OAuth2 required (consent screen needed)
- Scopes: calendar.readonly, calendar.events
- Rate limit: 1M queries/day (generous)
- REST API or googleapis npm package

### Simple Alternative: No-API Scheduling
- For MVP: use Supabase table as calendar
- No OAuth complexity, no external dependency
- Columns: event_name, start_time, end_time, attendees, created_by

## Code Patterns

### Supabase-only Calendar (MVP)
```sql
CREATE TABLE events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  title TEXT NOT NULL,
  start_at TIMESTAMPTZ NOT NULL,
  end_at TIMESTAMPTZ NOT NULL,
  attendees TEXT[],
  created_by UUID REFERENCES auth.users(id),
  metadata JSONB DEFAULT ''{}'',
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_events_time ON events (start_at, end_at);
```

### Time Slot Availability
```ts
async function getAvailableSlots(date: string, duration: number = 30) {
  const dayStart = new Date(date + "T09:00:00");
  const dayEnd = new Date(date + "T17:00:00");

  const { data: booked } = await supabase
    .from("events")
    .select("start_at, end_at")
    .gte("start_at", dayStart.toISOString())
    .lte("end_at", dayEnd.toISOString());

  // Generate all possible slots
  const slots = [];
  let current = dayStart;
  while (current < dayEnd) {
    const slotEnd = new Date(current.getTime() + duration * 60000);
    const isBooked = booked?.some(b =>
      new Date(b.start_at) < slotEnd && new Date(b.end_at) > current
    );
    if (!isBooked) slots.push({ start: current.toISOString(), end: slotEnd.toISOString() });
    current = slotEnd;
  }
  return slots;
}
```

## Known Gotchas
1. Timezone handling is the #1 source of calendar bugs — always store UTC, convert on display
2. Google Calendar OAuth requires verified app for >100 users
3. Recurring events are complex — start with one-time events only
4. Always validate: end_time > start_time, no double-booking
5. For MVP: skip Google Calendar integration, use Supabase-only approach',
  6.5
)
ON CONFLICT (module_id) DO NOTHING;

-- 6. AI as Product Feature (Claude API)
INSERT INTO zo_builder_modules (module_id, name, description, capabilities, content, quality_score)
VALUES (
  'bcm-ai-product',
  'AI as Product Feature (Claude API)',
  'Using Claude API as a product feature — matching, parsing, generation, analysis',
  ARRAY['ai-product', 'claude-api-feature', 'ai-matching', 'ai-parsing', 'text-analysis'],
  '# BCM: AI as Product Feature

## What This Module Provides
Patterns for using Claude API as a feature INSIDE products (not for building them).

## Architecture

### Edge Function Pattern (Recommended)
- User request → Next.js API → Supabase Edge Function → Claude API → response
- Edge Function handles API key securely
- Rate limit at API route level

### Direct API Route Pattern
- User request → Next.js API → Claude API → response
- Simpler, but API key in server environment

## Code Patterns

### AI Parsing (Natural Language → Structured Data)
```ts
// app/api/ai/parse/route.ts
import Anthropic from "@anthropic-ai/sdk";

const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

export async function POST(req: Request) {
  const { text, schema } = await req.json();

  const response = await anthropic.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 1024,
    system: `Extract structured data from user input. Return ONLY valid JSON matching this schema: ${JSON.stringify(schema)}`,
    messages: [{ role: "user", content: text }],
  });

  const parsed = JSON.parse(response.content[0].text);
  return Response.json(parsed);
}
```

### AI Matching (Semantic Similarity)
```ts
// Match user profile to opportunities
const response = await anthropic.messages.create({
  model: "claude-sonnet-4-20250514",
  max_tokens: 2048,
  system: "You are a matching engine. Score each opportunity 0-100 based on relevance to the user profile. Return JSON array sorted by score.",
  messages: [{
    role: "user",
    content: `Profile: ${JSON.stringify(userProfile)}\n\nOpportunities: ${JSON.stringify(opportunities)}`
  }],
});
```

### Cost Control
```ts
// Limit AI usage per user per day
const { count } = await supabase
  .from("ai_usage")
  .select("*", { count: "exact" })
  .eq("user_id", userId)
  .gte("created_at", todayStart);

if (count >= DAILY_LIMIT) {
  return Response.json({ error: "Daily AI limit reached" }, { status: 429 });
}
```

## Integration Points
- Supabase: store AI responses for caching (same input = cached output)
- Rate limiting: per-user daily caps (free: 10/day, pro: 100/day)
- Monitoring: log token usage to zo_cost_logs for budget tracking

## Known Gotchas
1. NEVER expose API key to client — always server-side
2. Implement response caching — same input should return cached result
3. Set max_tokens conservatively — users can trigger expensive queries
4. Always validate/sanitize user input before sending to Claude
5. Pricing: Claude Sonnet is ~$3/1M input tokens — at 1000 tokens/request, 1000 users/day = $3/day
6. Fallback: if Claude API is down, show cached results or graceful error',
  7.5
)
ON CONFLICT (module_id) DO NOTHING;

COMMIT;
