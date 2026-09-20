import AxeBuilder from '@axe-core/playwright'
import { expect, test } from '@playwright/test'

async function loginAsStudent(page) {
  await page.goto('/login')
  await page.getByRole('button', { name: /student/i }).click()
  await page.getByRole('button', { name: /^sign in/i }).click()
  await expect(page).toHaveURL(/\/dashboard/)
}

test.describe('login', () => {
  test('shows required-field errors and does not navigate', async ({ page }) => {
    await page.goto('/login')
    await page.getByRole('button', { name: /^sign in/i }).click()
    await expect(page.getByRole('alert')).toHaveCount(2)
    await expect(page).toHaveURL(/\/login/)
  })

  test('demo student can sign in with the keyboard alone', async ({ page }) => {
    await page.goto('/login')
    await page.getByLabel(/email/i).fill('student@sinhaspeech.lk')
    await page.keyboard.press('Tab')
    await page.keyboard.type('demo123')
    await page.keyboard.press('Enter')
    await expect(page).toHaveURL(/\/dashboard/)
  })

  test('unauthenticated visits to a protected page redirect to login', async ({ page }) => {
    await page.goto('/quizzes')
    await expect(page).toHaveURL(/\/login/)
  })
})

test.describe('role boundaries in the UI', () => {
  test('a student cannot open teacher pages', async ({ page }) => {
    await loginAsStudent(page)
    await page.goto('/teacher/quizzes')
    await expect(page).not.toHaveURL(/\/teacher\/quizzes/)
  })
})

test.describe('accessibility (axe, WCAG A/AA)', () => {
  const wcag = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']

  test('login page', async ({ page }) => {
    await page.goto('/login')
    const { violations } = await new AxeBuilder({ page }).withTags(wcag).analyze()
    expect(violations, JSON.stringify(violations.map((v) => v.id))).toEqual([])
  })

  for (const path of ['/dashboard', '/settings']) {
    test(`${path} page`, async ({ page }) => {
      await loginAsStudent(page)
      await page.goto(path)
      await page.waitForLoadState('networkidle')
      const { violations } = await new AxeBuilder({ page }).withTags(wcag).analyze()
      expect(violations, JSON.stringify(violations.map((v) => v.id))).toEqual([])
    })
  }

  test('high-contrast setting persists across reloads', async ({ page }) => {
    await loginAsStudent(page)
    await page.goto('/settings')
    await page.getByRole('checkbox').first().check({ force: true })
    await expect(page.locator('html')).toHaveAttribute('data-contrast', 'high')
    await page.reload()
    await expect(page.locator('html')).toHaveAttribute('data-contrast', 'high')
  })
})
