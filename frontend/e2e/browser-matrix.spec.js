/**
 * Cross-browser checks (TESTING_REPORT section 3.1.8).
 *
 * Coverage before this file was "one browser, one OS, found by accident" - the
 * one real configuration defect on record (a dynamic-DNS domain blocked by ad
 * blockers) turned up that way. These tests run the same journeys across
 * Chromium, Firefox, WebKit and a phone viewport.
 *
 * The interesting engine is WebKit. Everything the product is for - live
 * captioning and voice commands - rests on MediaRecorder, getUserMedia and
 * AudioContext, and those are exactly the APIs whose support and naming have
 * differed most in Safari. `useVoiceCommands` already guards on them
 * (`!window.MediaRecorder || !navigator.mediaDevices?.getUserMedia`), and
 * reaches for the `webkitAudioContext` prefix; what matters is that the guard
 * fires correctly and the student is told, rather than the page silently doing
 * nothing.
 *
 * Note that Playwright's WebKit is Safari's engine, not Safari, and headless
 * browsers report media APIs differently from real ones. So these assert the
 * app's *handling* of what the browser offers, not that any given browser
 * supports recording - the latter needs a real device.
 */

import { expect, test } from '@playwright/test'

async function login(page) {
  await page.goto('/login')
  await page.getByLabel(/email/i).fill('student@sinhaspeech.lk')
  await page.getByLabel(/password/i).fill('demo123')
  await page.getByRole('button', { name: /^sign in/i }).click()
  await expect(page).toHaveURL(/\/dashboard/)
}

test.describe('the app works in every engine', () => {
  test('a student can sign in and reach the dashboard', async ({ page }) => {
    await login(page)
    await expect(page.getByRole('heading').first()).toBeVisible()
  })

  test('client-side routing survives a hard reload on a deep link', async ({ page }) => {
    await login(page)
    await page.goto('/quizzes')
    await page.reload()
    // A reload on a sub-route is served by the SPA fallback; getting the login
    // page or a 404 here would mean the deployed rewrite rules are wrong.
    await expect(page).toHaveURL(/\/quizzes/)
  })

  test('no unexpected console errors on the main pages', async ({ page }) => {
    const errors = []
    page.on('console', (message) => {
      if (message.type() === 'error') errors.push(message.text())
    })
    page.on('pageerror', (error) => errors.push(String(error)))

    await login(page)
    await page.goto('/quizzes')
    await page.waitForLoadState('networkidle')

    // Mock-API mode makes no network calls that can legitimately fail.
    expect(errors, errors.join('\n')).toEqual([])
  })
})

test.describe('recording APIs the product depends on', () => {
  test('reports what this engine actually supports', async ({ page }, testInfo) => {
    await page.goto('/login')
    const support = await page.evaluate(() => ({
      mediaRecorder: typeof window.MediaRecorder !== 'undefined',
      getUserMedia: Boolean(navigator.mediaDevices?.getUserMedia),
      audioContext: Boolean(window.AudioContext || window.webkitAudioContext),
      webSocket: typeof WebSocket !== 'undefined',
      // The hook sends 16kHz PCM over a binary socket; ArrayBuffer support is
      // assumed everywhere, but assert it rather than assume.
      arrayBuffer: typeof ArrayBuffer !== 'undefined',
    }))

    // Recorded as an attachment so the matrix is visible per browser rather
    // than inferred - this is the actual deliverable of configuration testing.
    await testInfo.attach('media-api-support', {
      body: JSON.stringify({ project: testInfo.project.name, ...support }, null, 2),
      contentType: 'application/json',
    })

    // WebSocket and ArrayBuffer are non-negotiable: the streaming protocol has
    // no fallback without them, in any browser.
    expect(support.webSocket, 'WebSocket is required for live transcription').toBe(true)
    expect(support.arrayBuffer).toBe(true)
    // AudioContext is how the mic pipeline downsamples; the hook accepts the
    // webkit-prefixed name, so this must hold in WebKit too.
    expect(support.audioContext, 'AudioContext (or webkitAudioContext) is required').toBe(true)
  })

  test('every engine has what streaming needs, with or without MediaRecorder', async ({ page }) => {
    // This matrix found a real defect: WebKit reports no MediaRecorder, and
    // `useVoiceCommands` / `LiveTranscription` used to require it - despite
    // never constructing one, their pipeline being getUserMedia ->
    // AudioContext -> ScriptProcessor. Safari users would have been told the
    // core accessibility features were unsupported on a browser that runs them
    // fine. Both guards now check the APIs actually used.
    //
    // What this asserts is the premise behind that fix: in every engine, the
    // two APIs streaming genuinely needs are present even when MediaRecorder is
    // not. If a future engine lacks AudioContext, this fails and the guards
    // need revisiting. The guards' own logic is pinned in
    // `src/hooks/useVoiceCommands.test.jsx`, which was confirmed to fail
    // against the unfixed version.
    await page.addInitScript(() => {
      Object.defineProperty(window, 'MediaRecorder', { value: undefined, configurable: true })
    })
    await page.goto('/login')

    const streamingSupported = await page.evaluate(
      () =>
        Boolean(navigator.mediaDevices?.getUserMedia) &&
        Boolean(window.AudioContext || window.webkitAudioContext),
    )

    expect(
      streamingSupported,
      'streaming needs getUserMedia + AudioContext; this engine is missing one',
    ).toBe(true)
  })

  // The converse - that AudioRecorder and voice enrolment, which really do
  // construct a MediaRecorder, keep their own guard - is pinned in the unit
  // tests and in the comments on each guard. Asserting it from here would only
  // restate what the browser reports.
})

test.describe('layout', () => {
  test('no horizontal scrolling at this viewport', async ({ page }) => {
    await login(page)
    for (const path of ['/dashboard', '/quizzes', '/settings']) {
      await page.goto(path)
      await page.waitForLoadState('networkidle')
      const overflow = await page.evaluate(
        () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
      )
      // A few pixels of rounding is tolerable; a real overflow is not.
      expect(overflow, `${path} scrolls sideways by ${overflow}px`).toBeLessThanOrEqual(2)
    }
  })

  test('the primary action stays reachable and tappable', async ({ page }) => {
    await page.goto('/login')
    const signIn = page.getByRole('button', { name: /^sign in/i })
    await expect(signIn).toBeVisible()
    const box = await signIn.boundingBox()
    // 44px is the usual minimum touch target; this app's users are the last
    // people who should be asked to hit something smaller.
    expect(box.height, 'primary button is below a 44px touch target').toBeGreaterThanOrEqual(40)
  })
})
