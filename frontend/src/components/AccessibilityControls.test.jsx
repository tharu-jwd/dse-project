import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { axe } from 'jest-axe'
import { describe, expect, it } from 'vitest'
import AccessibilityControls from './AccessibilityControls'
import { AccessibilityProvider } from '../contexts/AccessibilityContext'
import { LanguageProvider } from '../contexts/LanguageContext'

const renderControls = (props) =>
  render(
    <LanguageProvider>
      <AccessibilityProvider>
        <AccessibilityControls {...props} />
      </AccessibilityProvider>
    </LanguageProvider>,
  )

describe('AccessibilityControls', () => {
  it('has no detectable WCAG violations (full and compact)', async () => {
    const full = renderControls()
    expect(await axe(full.container)).toHaveNoViolations()
    full.unmount()
    const compact = renderControls({ compact: true })
    expect(await axe(compact.container)).toHaveNoViolations()
  })

  it('exposes text size as a labelled group of toggle buttons', () => {
    renderControls()
    expect(screen.getByRole('group')).toBeInTheDocument()
    const buttons = screen.getAllByRole('button')
    expect(buttons).toHaveLength(3)
    buttons.forEach((b) => expect(b).toHaveAccessibleName())
    expect(buttons[0]).toHaveAttribute('aria-pressed', 'true')
  })

  it('changes text size by keyboard alone and persists it', async () => {
    const user = userEvent.setup()
    renderControls()
    const [, large] = screen.getAllByRole('button')
    await user.tab()
    await user.tab()
    expect(large).toHaveFocus()
    await user.keyboard('{Enter}')
    expect(large).toHaveAttribute('aria-pressed', 'true')
    expect(document.documentElement.dataset.transcriptSize).toBe('large')
    expect(JSON.parse(localStorage.getItem('sinhaspeech_accessibility')).fontSize).toBe('large')
  })

  it('toggles high contrast with the keyboard and reflects it on <html>', async () => {
    const user = userEvent.setup()
    renderControls()
    const checkbox = screen.getByRole('checkbox')
    checkbox.focus()
    await user.keyboard(' ')
    expect(checkbox).toBeChecked()
    expect(document.documentElement.dataset.contrast).toBe('high')
  })

  it('compact contrast button announces its pressed state', async () => {
    const user = userEvent.setup()
    renderControls({ compact: true })
    const button = screen.getByRole('button')
    expect(button).toHaveAttribute('aria-pressed', 'false')
    await user.click(button)
    expect(button).toHaveAttribute('aria-pressed', 'true')
  })

  it('restores saved preferences and survives corrupt storage', () => {
    localStorage.setItem('sinhaspeech_accessibility', JSON.stringify({ highContrast: true }))
    renderControls()
    expect(screen.getByRole('checkbox')).toBeChecked()
  })

  it('falls back to defaults when stored preferences are corrupt', () => {
    localStorage.setItem('sinhaspeech_accessibility', '{not json')
    renderControls()
    expect(screen.getByRole('checkbox')).not.toBeChecked()
  })
})
