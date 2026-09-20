/**
 * The quiz answer flow, driven by voice.
 *
 * This is the journey the whole product is for: a student who cannot
 * comfortably use a keyboard answers a quiz by speaking. The command handler in
 * QuizAnswerPage is where a recognised word becomes an action, and every one of
 * its guards protects against something a student would experience as the app
 * doing the wrong thing - skipping a question they had not answered, submitting
 * before they were ready, or acting twice on one spoken word.
 *
 * `useVoiceCommands` is mocked so tests can deliver a command directly. That
 * hook's own behaviour (sockets, mic, cleanup) is covered in
 * `src/hooks/useVoiceCommands.test.jsx`; what matters here is purely what the
 * page decides to do with a command once it arrives.
 */

import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { QuizAnswerPage } from './StudentQuizPages'
import { AccessibilityProvider } from '../contexts/AccessibilityContext'
import { LanguageProvider } from '../contexts/LanguageContext'

// The command callback the page registers, captured so tests can fire commands.
let sendCommand = () => {}
const voiceStop = vi.fn()
const voiceStart = vi.fn()

vi.mock('../hooks/useVoiceCommands', () => ({
  default: ({ onCommand }) => {
    sendCommand = (command) => onCommand?.(command)
    return {
      status: 'listening',
      error: '',
      voiceDetected: false,
      isListening: true,
      start: voiceStart,
      stop: voiceStop,
      registerBar: () => () => {},
    }
  },
  VOICE_METER_BAR_COUNT: 9,
}))

// Keeps the mic/upload machinery out of a test about command routing.
vi.mock('../components/useTranscriptionJob', () => ({
  default: () => ({ job: null, start: vi.fn(), reset: vi.fn() }),
}))
vi.mock('../components/AudioRecorder', () => ({ default: () => null }))
vi.mock('../components/LiveTranscription', () => ({ default: () => null }))
vi.mock('../components/TranscriptEditor', () => ({ default: () => null }))

const getQuiz = vi.fn()
const submitQuiz = vi.fn()
vi.mock('../api', () => ({
  api: {
    getQuiz: (...args) => getQuiz(...args),
    submitQuiz: (...args) => submitQuiz(...args),
    getTranscript: vi.fn(() => Promise.resolve({ id: 't1', segments: [] })),
    getSubmission: vi.fn(() => Promise.resolve({})),
  },
  API_BASE_URL: 'http://localhost:8000',
  USE_MOCK_API: true,
}))

const mcq = (id, text, optionCount = 4) => ({
  id,
  text,
  type: 'MCQ',
  required: true,
  options: Array.from({ length: optionCount }, (_, i) => ({
    id: `${id}-o${i + 1}`,
    text: `Option ${i + 1}`,
  })),
})

function quizOf(questions) {
  return {
    id: 'q1',
    title: 'Sinhala Listening Quiz',
    description: 'Answer by speaking.',
    submissionStatus: 'NOT_STARTED',
    questions,
  }
}

async function renderQuiz(quiz, { interactionMode = 'command' } = {}) {
  getQuiz.mockResolvedValue(quiz)
  localStorage.setItem('sinhaspeech_accessibility', JSON.stringify({ interactionMode }))

  render(
    <MemoryRouter initialEntries={['/quizzes/q1']}>
      <LanguageProvider>
        <AccessibilityProvider>
          <Routes>
            <Route path="/quizzes/:id" element={<QuizAnswerPage />} />
          </Routes>
        </AccessibilityProvider>
      </LanguageProvider>
    </MemoryRouter>,
  )
  // The page renders a loading state until getQuiz resolves.
  await screen.findByText(quiz.questions[0].text)
}

const optionButton = (n) => screen.getByRole('button', { name: new RegExp(`Option ${n}`) })
const isSelected = (n) => optionButton(n).className.includes('mcq-option--selected')
// Scoped to the question prompt: the confirm dialog also renders an h2.
const currentQuestion = () => document.querySelector('.question-prompt h2').textContent

beforeEach(() => {
  vi.clearAllMocks()
  localStorage.clear()
  submitQuiz.mockResolvedValue({})
})

describe('answering MCQ questions by voice', () => {
  it('selects the spoken option', async () => {
    await renderQuiz(quizOf([mcq('q-1', 'First question')]))

    expect(isSelected(2)).toBe(false)
    sendCommand('option_2')
    await waitFor(() => expect(isSelected(2)).toBe(true))
  })

  it('moves the selection rather than adding a second one', async () => {
    await renderQuiz(quizOf([mcq('q-1', 'First question')]))

    sendCommand('option_1')
    await waitFor(() => expect(isSelected(1)).toBe(true))
    sendCommand('option_3')

    await waitFor(() => expect(isSelected(3)).toBe(true))
    expect(isSelected(1)).toBe(false)
  })

  it('says so instead of selecting when the option does not exist', async () => {
    await renderQuiz(quizOf([mcq('q-1', 'First question', 2)]))

    sendCommand('option_4')

    expect(await screen.findByRole('status')).toBeInTheDocument()
    expect(isSelected(1)).toBe(false)
    expect(isSelected(2)).toBe(false)
  })

  it('clears the selection on "cancel"', async () => {
    await renderQuiz(quizOf([mcq('q-1', 'First question')]))

    sendCommand('option_2')
    await waitFor(() => expect(isSelected(2)).toBe(true))
    sendCommand('cancel')

    await waitFor(() => expect(isSelected(2)).toBe(false))
  })
})

describe('moving between questions by voice', () => {
  const twoQuestions = () => quizOf([mcq('q-1', 'First question'), mcq('q-2', 'Second question')])

  it('refuses to advance past an unanswered question', async () => {
    await renderQuiz(twoQuestions())

    sendCommand('next')

    expect(await screen.findByRole('status')).toBeInTheDocument()
    expect(currentQuestion()).toBe('First question')
  })

  it('advances once the question is answered', async () => {
    await renderQuiz(twoQuestions())

    sendCommand('option_1')
    await waitFor(() => expect(isSelected(1)).toBe(true))
    sendCommand('next')

    await waitFor(() => expect(currentQuestion()).toBe('Second question'))
  })

  it('goes back, and says so rather than wrapping around at the first question', async () => {
    await renderQuiz(twoQuestions())

    sendCommand('option_1')
    await waitFor(() => expect(isSelected(1)).toBe(true))
    sendCommand('next')
    await waitFor(() => expect(currentQuestion()).toBe('Second question'))

    sendCommand('previous')
    await waitFor(() => expect(currentQuestion()).toBe('First question'))

    sendCommand('previous')
    expect(await screen.findByRole('status')).toBeInTheDocument()
    expect(currentQuestion()).toBe('First question')
  })

  it('remembers the answer already given when returning to a question', async () => {
    await renderQuiz(twoQuestions())

    sendCommand('option_3')
    await waitFor(() => expect(isSelected(3)).toBe(true))
    sendCommand('next')
    await waitFor(() => expect(currentQuestion()).toBe('Second question'))
    sendCommand('previous')

    await waitFor(() => expect(currentQuestion()).toBe('First question'))
    expect(isSelected(3)).toBe(true)
  })
})

describe('submitting by voice', () => {
  const twoQuestions = () => quizOf([mcq('q-1', 'First question'), mcq('q-2', 'Second question')])

  async function reachTheEnd() {
    await renderQuiz(twoQuestions())
    sendCommand('option_1')
    await waitFor(() => expect(isSelected(1)).toBe(true))
    sendCommand('next')
    await waitFor(() => expect(currentQuestion()).toBe('Second question'))
    sendCommand('option_1')
    await waitFor(() => expect(isSelected(1)).toBe(true))
  }

  it('will not submit from a question that is not the last', async () => {
    await renderQuiz(twoQuestions())

    sendCommand('option_1')
    await waitFor(() => expect(isSelected(1)).toBe(true))
    sendCommand('submit')

    expect(await screen.findByRole('status')).toBeInTheDocument()
    expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument()
    expect(submitQuiz).not.toHaveBeenCalled()
  })

  it('will not submit while a required question is unanswered', async () => {
    await renderQuiz(twoQuestions())

    sendCommand('option_1')
    await waitFor(() => expect(isSelected(1)).toBe(true))
    sendCommand('next')
    await waitFor(() => expect(currentQuestion()).toBe('Second question'))

    sendCommand('submit') // last question, but this one is still unanswered

    expect(await screen.findByRole('status')).toBeInTheDocument()
    expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument()
    expect(submitQuiz).not.toHaveBeenCalled()
  })

  it('asks for confirmation first, and never submits on one spoken word', async () => {
    await reachTheEnd()

    sendCommand('submit')

    expect(await screen.findByRole('alertdialog')).toBeInTheDocument()
    expect(submitQuiz).not.toHaveBeenCalled()
  })

  it('submits every answer when the confirmation is spoken', async () => {
    await reachTheEnd()

    sendCommand('submit')
    await screen.findByRole('alertdialog')
    sendCommand('submit')

    await waitFor(() => expect(submitQuiz).toHaveBeenCalledTimes(1))
    expect(submitQuiz).toHaveBeenCalledWith('q1', [
      { questionId: 'q-1', selectedOptionId: 'q-1-o1' },
      { questionId: 'q-2', selectedOptionId: 'q-2-o1' },
    ])
  })

  it('"cancel" dismisses the confirmation without submitting', async () => {
    await reachTheEnd()

    sendCommand('submit')
    await screen.findByRole('alertdialog')
    sendCommand('cancel')

    await waitFor(() => expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument())
    expect(submitQuiz).not.toHaveBeenCalled()
  })

  it('swallows unrelated commands while the confirmation is open', async () => {
    await reachTheEnd()

    sendCommand('submit')
    await screen.findByRole('alertdialog')

    // "previous" must not navigate out from under an open confirmation.
    sendCommand('previous')

    expect(screen.getByRole('alertdialog')).toBeInTheDocument()
    expect(currentQuestion()).toBe('Second question')
    expect(submitQuiz).not.toHaveBeenCalled()
  })

  it('ignores further commands once the quiz has been submitted', async () => {
    await reachTheEnd()

    sendCommand('submit')
    await screen.findByRole('alertdialog')
    sendCommand('submit')
    await waitFor(() => expect(submitQuiz).toHaveBeenCalledTimes(1))

    sendCommand('submit')
    sendCommand('previous')

    await waitFor(() => expect(submitQuiz).toHaveBeenCalledTimes(1))
  })
})

describe('the same journey without voice', () => {
  const twoQuestions = () => quizOf([mcq('q-1', 'First question'), mcq('q-2', 'Second question')])

  it('can be completed by keyboard alone', async () => {
    const user = userEvent.setup()
    await renderQuiz(twoQuestions(), { interactionMode: 'normal' })

    await user.click(optionButton(1))
    await user.click(screen.getByRole('button', { name: /next/i }))
    await waitFor(() => expect(currentQuestion()).toBe('Second question'))

    await user.click(optionButton(2))
    await user.click(screen.getByRole('button', { name: /review|submit/i }))

    const dialog = await screen.findByRole('alertdialog')
    await user.click(within(dialog).getByRole('button', { name: /submit/i }))

    await waitFor(() => expect(submitQuiz).toHaveBeenCalledTimes(1))
  })

  it('keeps the next button disabled until the question is answered', async () => {
    await renderQuiz(twoQuestions(), { interactionMode: 'normal' })

    const next = screen.getByRole('button', { name: /next/i })
    expect(next).toBeDisabled()

    await userEvent.setup().click(optionButton(1))
    await waitFor(() => expect(next).toBeEnabled())
  })

  it('shows no voice-command feedback when voice is switched off', async () => {
    await renderQuiz(twoQuestions(), { interactionMode: 'normal' })

    sendCommand('next') // a stray command must not surface UI in this mode

    expect(screen.queryByRole('status')).not.toBeInTheDocument()
  })
})
