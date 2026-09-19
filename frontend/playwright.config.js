import { defineConfig, devices } from '@playwright/test'

// End-to-end tests run the real app in Chromium against the built-in mock API
// (VITE_USE_MOCK_API=true), so they need no backend. Kept out of CI by default:
// run with `npm run test:e2e`.
export default defineConfig({
  testDir: './e2e',
  reporter: 'list',
  use: { baseURL: 'http://localhost:4173', trace: 'retain-on-failure' },
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        // Set to reuse an already-installed Chromium instead of `npx playwright install`.
        launchOptions: process.env.PW_CHROMIUM_PATH
          ? { executablePath: process.env.PW_CHROMIUM_PATH }
          : {},
      },
    },
  ],
  webServer: {
    command: 'npm run build && npm run preview -- --port 4173 --strictPort',
    url: 'http://localhost:4173',
    env: { VITE_USE_MOCK_API: 'true' },
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
})
