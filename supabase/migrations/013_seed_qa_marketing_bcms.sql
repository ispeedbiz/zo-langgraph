-- Seed QA + Marketing BCMs for the 5 approved products
-- These give the Pipeline Architect domain-specific knowledge for downstream Minds
-- module_type = 'qa' for QA Mind, 'marketing' for Marketing Mind

BEGIN;

-- ===== QA BCMs (5) =====

-- 1. QA BCM: Equity & Education Email Products
INSERT INTO zo_builder_modules (module_id, name, description, capabilities, content, quality_score, module_type)
VALUES (
  'bcm-qa-equity-education',
  'QA: Equity & Education Email Products',
  'Test strategies for email newsletter products targeting finance professionals',
  ARRAY['qa-email-deliverability', 'qa-can-spam', 'qa-casl', 'qa-newsletter', 'qa-subscription-flow'],
  '# BCM-QA: Equity & Education Email Products

## What This Module Provides
QA strategies, test scenarios, and compliance requirements for email-based education products (newsletters, drip sequences, educational content delivery).

## Critical Compliance Tests

### CAN-SPAM (US)
- Every email MUST have physical mailing address
- Unsubscribe link MUST work within 10 business days
- Subject line MUST NOT be deceptive
- "From" field MUST accurately identify sender
- Test: Send test email → verify all 4 elements present
- Test: Click unsubscribe → verify removal within 1 request

### CASL (Canada — STRICT)
- Express consent REQUIRED before sending (no pre-checked boxes)
- Consent record MUST be stored (who, when, how, what was promised)
- Identify sender + provide contact info in every message
- Test: Sign up flow → verify explicit opt-in checkbox (unchecked by default)
- Test: Database → verify consent timestamp + method stored
- Test: Unsubscribe → verify processed within 10 days

### GDPR (if EU subscribers)
- Lawful basis for processing (consent)
- Right to erasure — delete all data on request
- Data portability — export subscriber data
- Test: Request data export → verify complete CSV generated
- Test: Request deletion → verify all records purged

## Deliverability Test Suite
- SPF record configured for sending domain
- DKIM signing enabled and verified
- DMARC policy set (at minimum p=none with reporting)
- Test with mail-tester.com score target: 9+/10
- Verify emails land in inbox (not spam) on: Gmail, Outlook, Yahoo
- Test HTML rendering across: Apple Mail, Gmail web, Outlook desktop
- Verify plain-text fallback exists and is readable
- Image alt-text present for all images
- Links all resolve (no 404s)
- Unsubscribe header (List-Unsubscribe) present

## Subscription Flow Tests
- Double opt-in: signup → confirmation email → click confirm → welcome email
- Welcome email sent within 60 seconds of confirmation
- Duplicate email prevention (same email cannot subscribe twice)
- Rate limiting on signup endpoint (prevent abuse)
- Honeypot or invisible CAPTCHA on signup form
- Test with special characters in email (name+tag@domain.com)
- Test with international domains (user@example.co.jp)

## Content Quality Tests
- Financial disclaimers present ("not financial advice", "for educational purposes")
- Sources cited for any market data or statistics
- Date/time stamps on market-sensitive content
- No broken chart/graph images
- Mobile responsive email template (test at 320px width)

## Edge Cases
- What happens when Resend API is down? (queue + retry, not lose email)
- What happens when subscriber list > 10,000? (batch sending)
- What happens when subscriber unsubscribes mid-drip sequence? (cancel remaining)
- Timezone handling for scheduled sends
- Bounce handling: soft bounce (retry 3x) vs hard bounce (auto-unsubscribe)',
  7.0,
  'qa'
);

-- 2. QA BCM: Grant Matching
INSERT INTO zo_builder_modules (module_id, name, description, capabilities, content, quality_score, module_type)
VALUES (
  'bcm-qa-grant-matching',
  'QA: Grant Matching & Government Data',
  'Test strategies for products that match users to government grants and funding',
  ARRAY['qa-gov-api', 'qa-eligibility-validation', 'qa-data-freshness', 'qa-grant-matching'],
  '# BCM-QA: Grant Matching & Government Data

## What This Module Provides
QA strategies for products that aggregate government grant data, match users to eligible grants, and handle sensitive eligibility criteria.

## Data Freshness Tests
- Grant data MUST be refreshed within 24 hours of source update
- Expired grants MUST NOT appear in search results
- Test: Insert grant with past deadline → verify excluded from results
- Test: Check last_synced timestamp → verify < 24 hours old
- Data source health check: if API returns stale data, flag it
- Version tracking: store raw API response hash, detect changes

## Government API Reliability Tests
- Rate limit handling: respect X-RateLimit headers, back off gracefully
- Timeout handling: 30 second max, retry with exponential backoff (max 3)
- API key rotation: test with expired key → verify graceful error + alert
- Test with API returning partial data (pagination edge case)
- Test with API returning 500 → verify cached data served with "last updated" warning
- Test with API format change (field renamed) → verify schema validation catches it

## Eligibility Validation Tests
- Business type filter: sole proprietor vs corporation vs nonprofit → different grants
- Province/state filter: Ontario grants NOT shown to BC users
- Revenue threshold: grants with "$500K max revenue" exclude larger businesses
- Industry code matching: NAICS codes correctly mapped
- Test: User with $600K revenue → verify $500K-cap grant excluded
- Test: User changes province → verify grant list refreshes
- Stacking rules: some grants cannot combine — verify mutual exclusion logic

## Search & Match Quality
- Relevance scoring: exact match > partial match > related
- Zero results: helpful message with broadening suggestions
- Test with misspellings (fuzzy matching threshold)
- Filter combinations: type + province + industry + revenue simultaneously
- Sort options: deadline (soonest first), amount (highest first), relevance
- Saved searches: user can save criteria and get notified of new matches

## Edge Cases
- Grant with "TBD" deadline — show but mark clearly
- Grant in French only (bilingual Canada requirement)
- User eligible for 200+ grants — pagination and performance
- Duplicate grants from different aggregation sources — deduplication
- Grant amount in range ("$5,000 - $50,000") — search both bounds',
  6.5,
  'qa'
);

-- 3. QA BCM: Invoice OCR & Intelligence
INSERT INTO zo_builder_modules (module_id, name, description, capabilities, content, quality_score, module_type)
VALUES (
  'bcm-qa-invoice-ocr',
  'QA: Invoice OCR & Intelligence',
  'Test strategies for products that parse, extract, and analyze invoice data',
  ARRAY['qa-ocr-accuracy', 'qa-invoice-parsing', 'qa-multi-format', 'qa-ifrs-compliance'],
  '# BCM-QA: Invoice OCR & Intelligence

## What This Module Provides
QA strategies for products that use OCR/AI to extract data from invoices, receipts, and financial documents.

## OCR Accuracy Tests
- Target accuracy: 95%+ on printed text, 85%+ on handwritten
- Test with: clean PDF, scanned image (300dpi), photo (phone camera), fax quality
- Test with: rotated documents (90°, 180°), skewed scans, partial page
- Multi-language: English, French (Canadian bilingual requirement)
- Currency symbols: $, CAD, USD, EUR, £ — all correctly parsed
- Number formats: 1,234.56 vs 1.234,56 (locale-aware)
- Date formats: MM/DD/YYYY, DD/MM/YYYY, YYYY-MM-DD — all parsed correctly

## Invoice Field Extraction Tests
- Required fields: vendor name, invoice number, date, total amount, tax
- Optional fields: line items, payment terms, PO number, due date
- Test: Invoice with all fields → verify 100% extraction
- Test: Invoice missing optional fields → verify no false data inserted
- Test: Invoice with multiple tax rates (HST 13% + PST 7%) → verify each parsed
- Test: Credit note (negative amounts) → verify sign preserved
- Confidence scores: each extracted field has confidence 0-100
- Low confidence (<70%) → flag for human review, do NOT auto-process

## Multi-Format Support
- PDF (native text extraction, no OCR needed)
- PDF (scanned/image — requires OCR)
- JPEG/PNG (photo of invoice)
- HEIC (iPhone photos)
- Excel/CSV (exported invoices)
- Test: Same invoice in all 5 formats → verify consistent extraction

## Accounting Compliance Tests
- IFRS: Revenue recognition rules — invoice date vs delivery date
- Tax calculation: extracted tax matches (subtotal × tax_rate)
- Currency conversion: if multi-currency, spot rate from date of invoice
- Duplicate detection: same vendor + same amount + similar date = flag
- Audit trail: every extraction logged with original document hash
- Data retention: configurable per jurisdiction (7 years Canada)

## Edge Cases
- Zero-amount invoice (proforma)
- Invoice with 100+ line items (pagination in PDF)
- Handwritten notes on printed invoice (ignore or flag?)
- Invoice in non-Latin script (Arabic numerals still extractable)
- Password-protected PDF → clear error message
- Corrupt file → graceful failure with specific error',
  6.5,
  'qa'
);

-- 4. QA BCM: Voice Interface Products
INSERT INTO zo_builder_modules (module_id, name, description, capabilities, content, quality_score, module_type)
VALUES (
  'bcm-qa-voice-interface',
  'QA: Voice Interface Products',
  'Test strategies for products with speech-to-text, voice commands, and audio interfaces',
  ARRAY['qa-speech-recognition', 'qa-voice-commands', 'qa-mic-permissions', 'qa-pwa-audio'],
  '# BCM-QA: Voice Interface Products

## What This Module Provides
QA strategies for products using Web Speech API, voice commands, and audio input/output.

## Speech Recognition Tests
- Browser support: Chrome (full), Safari (partial), Firefox (limited) — graceful degradation
- Microphone permission: first-time prompt → allow → recognition starts
- Microphone permission: denied → clear message explaining why mic is needed + how to re-enable
- Test in quiet environment: 95%+ accuracy on clear speech
- Test with background noise: 80%+ accuracy, or clear "noisy environment" warning
- Test with accents: Canadian English, Indian English, British English — all functional
- Test with numbers: "one hundred twenty three dollars and forty five cents" → $123.45
- Test with currency: "five hundred CAD" → 500.00 CAD

## Voice Command Tests
- Command recognition latency: < 500ms from end-of-speech to action
- Confirmation before destructive actions: "Delete invoice 47? Say yes to confirm"
- Undo support: "undo" reverses last voice action
- Help command: "help" or "what can I say" lists available commands
- Silence timeout: 5 seconds of silence → stop listening, show transcript
- Continuous listening vs push-to-talk: both modes work

## PWA Audio Tests
- Service worker does NOT intercept mic stream
- Offline mode: voice features gracefully disabled with message
- Background tab: mic stops when tab loses focus (privacy)
- Multiple tabs: only one tab captures mic at a time
- iOS Safari: Web Speech API limitations documented + fallback
- Android Chrome: full functionality verified

## Accessibility Tests
- Visual feedback while listening (animated indicator)
- Transcript always visible (deaf/HoH users can read what was heard)
- Keyboard alternative for every voice command
- Screen reader announces listening state changes
- High contrast mode: mic indicator still visible

## Edge Cases
- User says "period" or "comma" — insert punctuation or literal word?
- Very long dictation (>5 minutes) — memory and accuracy
- Bluetooth headset mic vs built-in mic switching mid-session
- Multiple speakers — warn "single speaker only"
- Non-English words in English speech (company names, foreign currencies)
- Browser tab crash during recording — recovery',
  6.5,
  'qa'
);

-- 5. QA BCM: Meeting Cost Calculator
INSERT INTO zo_builder_modules (module_id, name, description, capabilities, content, quality_score, module_type)
VALUES (
  'bcm-qa-meeting-calculator',
  'QA: Meeting Cost Calculator',
  'Test strategies for real-time meeting cost tracking and calendar integration products',
  ARRAY['qa-real-time-calculation', 'qa-timezone-handling', 'qa-calendar-sync', 'qa-cost-accuracy'],
  '# BCM-QA: Meeting Cost Calculator

## What This Module Provides
QA strategies for products that calculate meeting costs in real-time and integrate with calendar systems.

## Cost Calculation Accuracy Tests
- Salary to hourly rate: $100,000/year ÷ 2,080 hours = $48.08/hr — verify rounding
- Multiple attendees: 5 people × $48.08/hr × 1.5 hours = $360.60 — verify multiplication
- Real-time ticker: cost updates every second during meeting — verify smooth increment
- Overtime: meetings exceeding scheduled time → cost continues, visual warning
- Currency: display in user preferred currency, convert if attendees have different currencies
- Test with: minimum (2 people, 15 min), maximum (50 people, 8 hours)
- Verify: total cost = sum of individual costs (no floating point drift)

## Calendar Integration Tests
- Google Calendar: OAuth connect → list calendars → sync events
- Outlook/Microsoft: OAuth connect → list calendars → sync events
- Test: Create meeting in calendar → verify appears in app within 60 seconds
- Test: Update meeting time → verify cost recalculates
- Test: Cancel meeting → verify removed from cost tracking
- Test: Recurring meeting → each instance tracked separately
- Test: All-day event → excluded from cost calculation (or configurable)
- Permission scope: read-only calendar access (minimum required)

## Timezone Handling Tests
- Meeting at 3pm EST with attendee in PST → both see correct local time
- Daylight saving transition: meeting scheduled for day of DST change → correct
- Test: User in UTC+5:30 (India) → half-hour offset handled
- Test: Meeting spanning midnight → date handled correctly
- Display: always show user local time with timezone indicator

## Real-Time Performance Tests
- Timer accuracy: after 1 hour, drift < 1 second
- Browser tab inactive: timer pauses or catches up? (configurable)
- Multiple simultaneous meetings: each tracks independently
- Page refresh during meeting: timer resumes from correct position
- Network disconnection: timer continues locally, syncs when reconnected

## Edge Cases
- Meeting with no salary data for some attendees → use company average or exclude?
- Attendee joins late / leaves early → partial cost tracking
- External attendees (no salary data) → configurable placeholder rate
- Meeting cost exceeds budget threshold → notification to organizer
- Historical reports: meeting costs over time, trends, department breakdown
- Privacy: individual salaries never exposed, only aggregate cost shown',
  6.5,
  'qa'
);

-- ===== MARKETING BCMs (5) =====

-- 1. Marketing BCM: Equity & Education
INSERT INTO zo_builder_modules (module_id, name, description, capabilities, content, quality_score, module_type)
VALUES (
  'bcm-mkt-equity-education',
  'Marketing: Equity & Education Products',
  'Audience, channels, and messaging strategy for finance education email products',
  ARRAY['mkt-finance-professionals', 'mkt-linkedin-authority', 'mkt-trust-messaging', 'mkt-newsletter-growth'],
  '# BCM-MKT: Equity & Education Products

## What This Module Provides
Marketing strategy, audience definition, channels, and messaging hooks for email-based finance education products.

## Target Audience
### Primary: Finance Professionals Seeking Edge
- CAs, CPAs, CFOs, financial analysts (25-55 years)
- Pain: information overload, no time to curate quality learning
- Want: curated, actionable insights delivered to inbox
- Where they live: LinkedIn, finance podcasts, CPA community forums
- Trust signals: credentials matter (CA, CPA, CFA after name = credibility)

### Secondary: Business Owners Managing Finances
- Founders, SMB owners wearing the "finance hat"
- Pain: dont understand equity implications of decisions
- Want: simple explanations, not textbook language
- Where they live: Twitter/X, startup communities, Reddit r/smallbusiness

## Channel Strategy
### LinkedIn (PRIMARY — 60% of effort)
- Authority content: weekly insight posts from product knowledge base
- Format: hook → insight → "subscribe for weekly deep dives" CTA
- Target: finance hashtags (#accounting #cpa #cfo #financialplanning)
- Engagement: comment on industry posts with genuine insights + soft mention

### Email (RETENTION — 30% of effort)
- Welcome sequence: 3 emails over 7 days (value → social proof → habit)
- Email 1 (immediate): "Here is your first insight" — deliver value instantly
- Email 2 (day 3): "3,000 finance pros read this weekly" — social proof
- Email 3 (day 7): "Your weekly ritual starts now" — habit formation
- Subject line formulas: number + specific + benefit ("5 equity ratios CFOs check every Monday")
- Unsubscribe rate target: <0.5% per email

### Product Hunt (LAUNCH — 10% of effort)
- Category: Productivity, Finance
- Tagline: "Equity intelligence delivered weekly — written by AI, curated by a CA"
- First comment: founder story (CA who built AI to teach what took years to learn)

## Messaging Hooks
- "Stop Googling equity concepts. Start understanding them."
- "The weekly email 3,000 finance pros actually read."
- "Written by AI. Curated by a Chartered Accountant."
- "5-minute reads. Career-long value."

## Trust Architecture
- Free tier generous (builds trust before asking for anything)
- Founder credentials prominently displayed (CA designation)
- Testimonials: "I learned more in 4 weeks than in my CPA prep course"
- Unsubscribe easy and guilt-free (builds trust, reduces spam reports)
- Privacy: "We never share your email. Ever."',
  7.0,
  'marketing'
);

-- 2. Marketing BCM: Grant Matching
INSERT INTO zo_builder_modules (module_id, name, description, capabilities, content, quality_score, module_type)
VALUES (
  'bcm-mkt-grant-matching',
  'Marketing: Grant Matching Products',
  'Audience, channels, and messaging strategy for grant discovery and matching products',
  ARRAY['mkt-nonprofit-audience', 'mkt-startup-funding', 'mkt-urgency-deadlines', 'mkt-grant-discovery'],
  '# BCM-MKT: Grant Matching Products

## What This Module Provides
Marketing strategy for products that help users discover and apply for government grants and funding.

## Target Audience
### Primary: Small Business Owners (Canada focus)
- Sole proprietors, micro-businesses (1-10 employees)
- Pain: "I know grants exist but I cannot find ones I qualify for"
- Want: personalized matches without reading 200 government pages
- Income: $50K-$500K revenue, grants could be 10-50% of annual revenue
- Where they live: local business Facebook groups, BDC events, accountant referrals

### Secondary: Nonprofit Organizations
- Program managers, executive directors
- Pain: grant applications are full-time jobs, miss deadlines
- Want: automatic alerts when new relevant grants appear
- Where they live: CharityVillage, nonprofit LinkedIn groups, sector associations

### Tertiary: Startup Founders
- Pre-seed to Series A, especially deeptech/cleantech
- Pain: "We need non-dilutive funding but dont know where to look"
- Want: SR&ED, IRAP, provincial innovation grants matched automatically
- Where they live: startup Slack communities, YC forums, Twitter/X tech

## Channel Strategy
### SEO (PRIMARY — 50% of effort)
- Target keywords: "small business grants Canada 2026", "Ontario grants for startups"
- Content: monthly updated grant lists (high search volume, high intent)
- Each grant page: eligibility checker tool (interactive, keeps users on site)
- Long-tail: "grants for women-owned businesses Ontario" (less competition)

### Partnerships (SECONDARY — 30% of effort)
- Accountants: referral program ("your clients are leaving money on the table")
- Business Development Bank of Canada (BDC): content partnership
- Local chambers of commerce: free grant webinars → product demo

### Social Proof (ONGOING — 20% of effort)
- Success stories: "$47,000 in grants found in 3 minutes"
- Counter on homepage: "Total grants matched: $X,XXX,XXX"
- Deadline urgency: "37 grants closing this month — check eligibility now"

## Messaging Hooks
- "You are leaving free money on the table."
- "$47,000 in grants. Found in 3 minutes. Zero dilution."
- "The grant matching engine that never misses a deadline."
- "Stop reading government websites. Start getting matched."

## Urgency Mechanics
- Deadline countdown: "IRAP closes in 12 days — check eligibility"
- New grant alerts: "3 new grants match your profile this week"
- Seasonal pushes: Q1 (new fiscal year grants), Q4 (use-it-or-lose-it budgets)',
  7.0,
  'marketing'
);

-- 3. Marketing BCM: Invoice Intelligence
INSERT INTO zo_builder_modules (module_id, name, description, capabilities, content, quality_score, module_type)
VALUES (
  'bcm-mkt-invoice-intelligence',
  'Marketing: Invoice Intelligence Products',
  'Audience, channels, and messaging strategy for invoice OCR and automation products',
  ARRAY['mkt-small-business', 'mkt-data-entry-pain', 'mkt-roi-calculator', 'mkt-invoice-automation'],
  '# BCM-MKT: Invoice Intelligence Products

## What This Module Provides
Marketing strategy for products that automate invoice data extraction, categorization, and accounting integration.

## Target Audience
### Primary: Small Business Owners (1-50 employees)
- Bookkeepers, office managers doing manual data entry
- Pain: "I spend 8 hours a week typing invoice numbers into QuickBooks"
- Want: snap a photo → data extracted → posted to accounting software
- Price sensitivity: high — must show clear ROI within first month
- Where they live: QuickBooks community, Xero forums, bookkeeper Facebook groups

### Secondary: Accounting Firms
- Staff accountants processing client invoices
- Pain: "Every client sends invoices in different formats"
- Want: one tool that handles PDF, scan, photo, email attachment
- Volume: 200-2000 invoices/month per firm
- Where they live: CPA Canada events, accounting technology conferences

## Channel Strategy
### Product-Led Growth (PRIMARY — 50% of effort)
- Free tier: 20 invoices/month (enough to prove value)
- Onboarding: first invoice scanned in under 60 seconds (aha moment)
- Upgrade trigger: "You have processed 18/20 free invoices this month"
- Referral program: "Give 10 free invoices, get 10 free invoices"

### Content Marketing (SECONDARY — 30% of effort)
- ROI calculator: "Enter your hourly rate + invoices per month = time saved"
- Blog: "The true cost of manual data entry" (searchable, shareable)
- Video: 30-second demo — photo of invoice → extracted data → QuickBooks
- Comparison pages: "InvoiceMemory vs Dext vs HubDoc" (capture competitor searches)

### Accounting Software Marketplaces (LAUNCH — 20% of effort)
- QuickBooks App Store listing
- Xero App Marketplace listing
- FreshBooks integration directory
- These marketplaces have built-in audiences actively seeking automation

## Messaging Hooks
- "Stop typing. Start scanning."
- "8 hours of data entry → 8 minutes."
- "Your phone camera is now your best bookkeeper."
- "AI reads your invoices so you dont have to."

## ROI Messaging
- Time saved: average 6.5 hours/week for a 200-invoice/month business
- Error reduction: 95% fewer data entry mistakes
- Payback: subscription pays for itself in the first week
- Compound value: categorization AI gets smarter with every invoice',
  7.0,
  'marketing'
);

-- 4. Marketing BCM: Voice Interface Products
INSERT INTO zo_builder_modules (module_id, name, description, capabilities, content, quality_score, module_type)
VALUES (
  'bcm-mkt-voice-interface',
  'Marketing: Voice Interface Products',
  'Audience, channels, and messaging strategy for voice-first and hands-free products',
  ARRAY['mkt-accessibility', 'mkt-field-workers', 'mkt-hands-free', 'mkt-voice-demo'],
  '# BCM-MKT: Voice Interface Products

## What This Module Provides
Marketing strategy for products with voice-first interfaces — hands-free invoicing, voice commands, speech-to-data.

## Target Audience
### Primary: Field Service Workers
- Contractors, plumbers, electricians, cleaners
- Pain: "I finish a job and forget to invoice. Or I do it at midnight."
- Want: say "Invoice John Smith, replaced kitchen faucet, $350" → done
- Context: hands dirty, in a van, phone in pocket — CANNOT type
- Where they live: trade Facebook groups, contractor forums, Google Maps reviews

### Secondary: Accessibility-First Users
- Users with mobility impairments, repetitive strain injury, visual impairments
- Pain: traditional invoice software requires mouse + keyboard + reading
- Want: fully voice-operated workflow from creation to sending
- Where they live: accessibility communities, assistive tech forums

### Tertiary: Mobile-First Entrepreneurs
- Food truck owners, market vendors, freelance photographers
- Pain: no desk, no computer, business runs from phone
- Want: create and send invoices without sitting down
- Where they live: Instagram, TikTok (small business content), Etsy seller forums

## Channel Strategy
### Video Demos (PRIMARY — 40% of effort)
- 15-second TikTok/Reels: person in work truck says invoice → phone creates it
- 60-second YouTube: full workflow — voice command → invoice → client receives email
- Before/after: "Old way: 15 minutes typing" vs "New way: 15 seconds talking"
- Accessibility demo: screen reader user creating invoice entirely by voice

### Trade Communities (SECONDARY — 30% of effort)
- Facebook groups for contractors/tradespeople (organic posts, not ads)
- Home service platforms: Thumbtack, HomeAdvisor, TaskRabbit
- Trade school partnerships: teach students modern invoicing
- Local chamber of commerce: "Tech tools for tradespeople" workshops

### App Store Optimization (LAUNCH — 30% of effort)
- PWA listing emphasis: "Works on any phone, no app download needed"
- Keywords: "voice invoice", "hands free invoicing", "speak to invoice"
- Screenshots showing voice waveform → completed invoice

## Messaging Hooks
- "Say it. Send it. Get paid."
- "Your voice is your new invoice software."
- "Hands dirty? Voice clean. Invoice done."
- "The invoice app that listens."

## Demo Strategy
- Interactive demo on landing page: click mic → say test invoice → see it created
- No signup required for demo (reduce friction)
- Demo data auto-populates with realistic trade examples',
  6.5,
  'marketing'
);

-- 5. Marketing BCM: Meeting Cost Products
INSERT INTO zo_builder_modules (module_id, name, description, capabilities, content, quality_score, module_type)
VALUES (
  'bcm-mkt-meeting-calculator',
  'Marketing: Meeting Cost Products',
  'Audience, channels, and messaging strategy for meeting cost tracking and calendar analytics',
  ARRAY['mkt-manager-executive', 'mkt-viral-calculator', 'mkt-meeting-culture', 'mkt-productivity'],
  '# BCM-MKT: Meeting Cost Products

## What This Module Provides
Marketing strategy for products that calculate real-time meeting costs and provide calendar analytics.

## Target Audience
### Primary: Middle Managers & Team Leads
- Managing 5-20 person teams in tech, consulting, finance
- Pain: "Half my day is meetings and I suspect most are wasteful"
- Want: data to justify declining/shortening meetings
- Where they live: LinkedIn, Manager Tools podcast, Lara Hogan blog, HBR

### Secondary: CFOs & Operations Leaders
- Responsible for workforce productivity metrics
- Pain: "I know meetings are expensive but I cannot quantify it"
- Want: company-wide meeting cost dashboard, trend reports
- Where they live: CFO conferences, Gartner reports, finance LinkedIn

### Tertiary: Individual Contributors
- Engineers, designers, writers who feel meeting-overloaded
- Pain: "I had 6 hours of meetings today and zero coding time"
- Want: personal meeting load tracker, time-in-meetings vs time-creating ratio
- Where they live: Hacker News, Dev.to, Twitter/X tech, r/programming

## Channel Strategy
### Viral Calculator Widget (PRIMARY — 50% of effort)
- Free embeddable widget: "How much do your meetings cost?"
- Input: number of attendees, average salary, meeting length → shocking total
- Share button: "My last all-hands cost $4,700" → Twitter/LinkedIn share
- This IS the marketing — the product markets itself through shock value
- Track: shares, embeds, click-through to full product

### LinkedIn Thought Leadership (SECONDARY — 30% of effort)
- Weekly stats: "Average company with 100 employees spends $2.1M/year on meetings"
- Polls: "How many hours were you in meetings this week?" (engagement bait)
- Carousel: "5 meetings that should have been emails (with cost breakdown)"
- Tag executives and productivity authors for amplification

### Product Hunt & Tech Press (LAUNCH — 20% of effort)
- Category: Productivity
- Angle: "The meeting cost calculator your CFO needs to see"
- Press pitch: "Startup builds tool that calculates the real cost of your meetings in real-time"
- Data angle: "We analyzed 10,000 meetings — here is what we found" (aggregate anonymized data)

## Messaging Hooks
- "Your meetings cost more than you think."
- "That status update just cost $340."
- "The meeting expense your company never tracks."
- "Make meetings accountable. Make them count."

## Viral Mechanics
- Shock value: large dollar amounts create emotional response → sharing
- Personal relevance: "YOUR meetings cost THIS much" — makes it personal
- Social proof: counter showing "X meetings tracked globally"
- Competitive: teams compare meeting costs → drives adoption within companies
- Slack/Teams integration: "/meetingcost" command → cost in real-time in chat',
  7.0,
  'marketing'
);

COMMIT;
