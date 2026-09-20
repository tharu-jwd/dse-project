import { render } from '@testing-library/react'
import { axe } from 'jest-axe'
import { describe, expect, it, vi } from 'vitest'
import VoiceMeter from './VoiceMeter'
import { VOICE_METER_BAR_COUNT } from '../hooks/useVoiceCommands'

describe('VoiceMeter', () => {
  it('renders one bar per meter slot and registers each with the hook', () => {
    const register = vi.fn(() => () => {})
    const { container } = render(<VoiceMeter registerBar={register} active={false} />)
    expect(container.querySelectorAll('i')).toHaveLength(VOICE_METER_BAR_COUNT)
    expect(register).toHaveBeenCalledTimes(VOICE_METER_BAR_COUNT)
  })

  it('reflects active and compact state as classes and has no a11y violations', async () => {
    const { container } = render(<VoiceMeter registerBar={() => () => {}} active compact />)
    expect(container.firstChild).toHaveClass('voice-meter--active', 'voice-meter--compact')
    expect(await axe(container)).toHaveNoViolations()
  })
})
