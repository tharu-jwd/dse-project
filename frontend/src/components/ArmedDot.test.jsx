import { act, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import ArmedDot from './ArmedDot'

const announce = (seconds) =>
  act(() => {
    window.dispatchEvent(new CustomEvent('voice-armed', { detail: { seconds } }))
  })

describe('ArmedDot', () => {
  afterEach(() => vi.useRealTimers())

  it('is hidden until the wake word is heard', () => {
    const { container } = render(<ArmedDot />)
    expect(container.querySelector('.armed-dot')).toBeNull()
  })

  it('appears when armed and vanishes when the wake window ends', () => {
    vi.useFakeTimers()
    const { container } = render(<ArmedDot />)
    announce(20)
    expect(container.querySelector('.armed-dot')).not.toBeNull()
    act(() => vi.advanceTimersByTime(20_000))
    expect(container.querySelector('.armed-dot')).toBeNull()
  })

  it('vanishes as soon as a command uses the window up', () => {
    const { container } = render(<ArmedDot />)
    announce(20)
    announce(0)
    expect(container.querySelector('.armed-dot')).toBeNull()
  })
})
