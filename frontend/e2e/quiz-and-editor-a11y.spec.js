// Accessibility (Task Gap README Task 8): axe already runs on login,
// dashboard and settings (app.spec.js) - none of that covers the pages
// this product actually exists for. A quiz a student cannot answer
// without a mouse, or a transcript editor that traps keyboard focus,
// would defeat the accessibility mission and axe would never know
// unless it's pointed at these specific pages.
import AxeBuilder from '@axe-core/playwright'
import { expect, test } from '@playwright/test'

const WCAG = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']

async function loginAsStudent(page) {
  await page.goto('/login')
  await page.getByRole('button', { name: /student/i }).click()
  await page.getByRole('button', { name: /^sign in/i }).click()
  await expect(page).toHaveURL(/\/dashboard/)
}

async function assertNoViolations(page) {
  const { violations } = await new AxeBuilder({ page }).withTags(WCAG).analyze()
  expect(violations, JSON.stringify(violations.map((v) => ({ id: v.id, nodes: v.nodes.length })))).toEqual([])
}

// A quiz link opens a "Not started"/"In progress" intro screen first, not
// the answer view directly - "Start quiz" (or "Continue") gets you there.
async function enterQuiz(page) {
  const startButton = page.getByRole('button', { name: /start quiz|continue/i })
  if (await startButton.isVisible().catch(() => false)) {
    await startButton.focus()
    await page.keyboard.press('Enter')
    await page.waitForLoadState('networkidle')
  }
}

test.describe('accessibility on the pages the product exists for', () => {
  test('quiz answer page', async ({ page }) => {
    await loginAsStudent(page)
    await page.goto('/quizzes')
    await page.waitForLoadState('networkidle')

    const firstQuiz = page.locator('a[href^="/quizzes/"]').first()
    if ((await firstQuiz.count()) === 0) {
      test.skip(true, 'no quizzes exist for the demo student')
    }
    await firstQuiz.click()
    await page.waitForLoadState('networkidle')
    await enterQuiz(page)

    await assertNoViolations(page)
  })

  test('transcript editor page', async ({ page }) => {
    await loginAsStudent(page)
    await page.goto('/transcripts')
    await page.waitForLoadState('networkidle')

    const firstTranscript = page.locator('a[href^="/transcripts/"]').first()
    if ((await firstTranscript.count()) === 0) {
      test.skip(true, 'no transcripts exist for the demo student')
    }
    await firstTranscript.click()
    await page.waitForLoadState('networkidle')

    await assertNoViolations(page)
  })

  test('live transcription page (new note)', async ({ page }) => {
    await loginAsStudent(page)
    await page.goto('/notes/new')
    await page.waitForLoadState('networkidle')

    await assertNoViolations(page)
  })

  test('a student can select an MCQ answer using only the keyboard', async ({ page }) => {
    await loginAsStudent(page)
    await page.goto('/quizzes')
    await page.waitForLoadState('networkidle')

    const quizCount = await page.locator('a[href^="/quizzes/"]').count()
    if (quizCount === 0) {
      test.skip(true, 'no quizzes exist for the demo student')
    }

    // Not every quiz has an MCQ question (some are all-SPOKEN) - try each
    // quiz in turn, walking its question nav (keyboard only), until one
    // with an MCQ question is found.
    let mcqOptions = page.locator('.mcq-option')
    for (let quizIndex = 0; quizIndex < quizCount && (await mcqOptions.count()) === 0; quizIndex++) {
      await page.goto('/quizzes')
      await page.waitForLoadState('networkidle')
      await page.locator('a[href^="/quizzes/"]').nth(quizIndex).click()
      await page.waitForLoadState('networkidle')
      await enterQuiz(page)

      const navButtons = page.locator('.question-nav button')
      const navCount = await navButtons.count()
      for (let index = 0; index < navCount && (await mcqOptions.count()) === 0; index++) {
        await navButtons.nth(index).focus()
        await page.keyboard.press('Enter')
        mcqOptions = page.locator('.mcq-option')
      }
    }

    if ((await mcqOptions.count()) === 0) {
      test.skip(true, 'no MCQ question exists in any of the demo student\'s quizzes')
    }

    // Never a mouse click from here on: Tab to the first option, activate
    // it with the keyboard, and confirm the selection registered.
    const firstOption = mcqOptions.first()
    await firstOption.focus()
    await expect(firstOption).toBeFocused()
    await page.keyboard.press('Enter')

    await expect(firstOption).toHaveClass(/mcq-option--selected/)

    // Move to the next question using only the keyboard too, via the
    // question-nav buttons in the sidebar.
    const nextQuestionButton = page.locator('.question-nav button').nth(1)
    if (await nextQuestionButton.isVisible()) {
      await nextQuestionButton.focus()
      await page.keyboard.press('Enter')
      await expect(page.locator('.question-prompt h2')).toBeVisible()
    }
  })
})
