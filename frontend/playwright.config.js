import { defineConfig, devices } from '@playwright/test'

// End-to-end tests run the real app against the built-in mock API
// (VITE_USE_MOCK_API=true), so they need no backend. Kept out of CI by default:
// run with `npm run test:e2e`.
//
// Browser matrix (TESTING_REPORT section 3.1.8): the app's recording features
// depend on MediaRecorder, getUserMedia and AudioContext, whose support has
// historically differed most between Chromium and WebKit - so WebKit is the
// engine worth caring about here, not an afterthought.
//
// One honest limit: Playwright's WebKit is Safari's *engine*, not Safari. It is
// close enough to catch API-shape differences and layout bugs, and not close
// enough to certify Safari on iOS. Real-device checks are still owed.
//
//   npm run test:e2e                  all installed browsers
//   npm run test:e2e -- --project=webkit
// Both Chromium-backed projects share this. PW_CHROMIUM_PATH points them at an
// already-installed Chromium when `npx playwright install chromium` cannot
// complete - the download stalled repeatedly on at least one machine here, and
// a browser matrix that cannot be run is worth nothing. `channel: undefined`
// keeps Playwright from preferring its own bundled build over that path.
const chromium = process.env.PW_CHROMIUM_PATH
  ? { launchOptions: { executablePath: process.env.PW_CHROMIUM_PATH }, channel: undefined }
  : {}

export default defineConfig({
  testDir: './e2e',
  reporter: 'list',
  // Four browser projects on one machine is genuinely slow: each test pays for
  // a browser launch, and Firefox in particular can take tens of seconds to
  // start when the machine is busy. Raised from the 30s default because that
  // produced timeouts that said nothing about the application. Retries stay at
  // zero deliberately - a retry would hide a real intermittent bug, which on an
  // accessibility-critical path is the last thing worth hiding.
  timeout: 90_000,
  use: { baseURL: 'http://localhost:4173', trace: 'retain-on-failure' },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'], ...chromium } },
    { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
    { name: 'webkit', use: { ...devices['Desktop Safari'] } },
    // A phone viewport, since students may well be on one. Chromium-backed so
    // this stays about layout rather than a second engine's quirks.
    { name: 'mobile-chrome', use: { ...devices['Pixel 7'], ...chromium } },
  ],
  webServer: {
    command: 'npm run build && npm run preview -- --port 4173 --strictPort',
    url: 'http://localhost:4173',
    env: { VITE_USE_MOCK_API: 'true' },
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
})
