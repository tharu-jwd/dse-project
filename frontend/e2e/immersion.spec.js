// "Customer immersion" check (Task Gap README Task 6): visit every
// student-facing page and fail if anything internal leaked onto the
// screen - a stringified error, a raw backend id, an unformatted
// timestamp, or a field name a student was never meant to see. None of
// the other suites check the *rendered text itself* against this list;
// axe checks structure/contrast, the API tests check status codes, but
// nothing has ever grepped the DOM for "undefined" or a bare UUID.
import { expect, test } from '@playwright/test'

const LEAKED_TOKENS = [
  'undefined',
  'null',
  'NaN',
  '[object Object]',
  'Traceback',
  'Internal Server Error',
  'transcriptId',
  'isCorrect',
]

// 8-4-4-4-12 hex, the shape of every id this app hands out (user_id,
// transcript_id, quiz_id, ...) - a student should only ever see a
// human title, never this.
const RAW_UUID = /\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b/i

// An ISO 8601 timestamp with the 'T' separator still in it - every
// formatted date in this app should have gone through a human-readable
// formatter (date-fns / toLocaleString) before render.
const RAW_ISO_TIMESTAMP = /\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/

async function loginAsStudent(page) {
  await page.goto('/login')
  await page.getByRole('button', { name: /student/i }).click()
  await page.getByRole('button', { name: /^sign in/i }).click()
  await expect(page).toHaveURL(/\/dashboard/)
}

function assertNoLeaks(text, where) {
  for (const token of LEAKED_TOKENS) {
    expect(text.includes(token), `${where}: found leaked token ${JSON.stringify(token)}`).toBe(false)
  }

  const uuidMatch = text.match(RAW_UUID)
  expect(uuidMatch, `${where}: found a raw UUID (${uuidMatch?.[0]}) in visible text`).toBeNull()

  const isoMatch = text.match(RAW_ISO_TIMESTAMP)
  expect(isoMatch, `${where}: found an unformatted ISO timestamp (${isoMatch?.[0]}) in visible text`).toBeNull()
}

test.describe('nothing internal leaks to the student screen', () => {
  test('login page (pre-auth)', async ({ page }) => {
    await page.goto('/login')
    assertNoLeaks(await page.locator('body').innerText(), '/login')
  })

  const staticPages = ['/dashboard', '/transcripts', '/quizzes', '/settings', '/help']

  for (const path of staticPages) {
    test(`${path} page`, async ({ page }) => {
      await loginAsStudent(page)
      await page.goto(path)
      await page.waitForLoadState('networkidle')
      assertNoLeaks(await page.locator('body').innerText(), path)
    })
  }

  test('first transcript detail page, if any exist', async ({ page }) => {
    await loginAsStudent(page)
    await page.goto('/transcripts')
    await page.waitForLoadState('networkidle')

    const firstLink = page.locator('a[href^="/transcripts/"]').first()
    if ((await firstLink.count()) === 0) {
      test.skip(true, 'no transcripts exist for the demo student')
    }

    await firstLink.click()
    await page.waitForLoadState('networkidle')
    assertNoLeaks(await page.locator('body').innerText(), 'transcript detail')
  })

  test('first quiz answer page, if any exist', async ({ page }) => {
    await loginAsStudent(page)
    await page.goto('/quizzes')
    await page.waitForLoadState('networkidle')

    const firstLink = page.locator('a[href^="/quizzes/"]').first()
    if ((await firstLink.count()) === 0) {
      test.skip(true, 'no quizzes exist for the demo student')
    }

    await firstLink.click()
    await page.waitForLoadState('networkidle')
    assertNoLeaks(await page.locator('body').innerText(), 'quiz answer page')
  })
})
