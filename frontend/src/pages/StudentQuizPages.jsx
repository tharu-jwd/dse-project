import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api } from '../api'
import quizBackground from '../assets/4.jpg'
import AudioRecorder from '../components/AudioRecorder'
import Icon from '../components/Icon'
import TranscriptEditor from '../components/TranscriptEditor'
import TranscriptionStatus from '../components/TranscriptionStatus'
import useTranscriptionJob from '../components/useTranscriptionJob'
import useVoiceCommands from '../hooks/useVoiceCommands'
import VoiceMeter from '../components/VoiceMeter'
import { useAccessibility } from '../contexts/AccessibilityContext'
import {
  Alert,
  ConfirmDialog,
  EmptyState,
  Loading,
  PageHeader,
  ProgressSteps,
  StatusBadge,
} from '../components/UI'

export function QuizListPage() {
  const [quizzes, setQuizzes] = useState(null)
  const [error, setError] = useState('')
  useEffect(() => {
    api
      .getQuizzes()
      .then(setQuizzes)
      .catch((cause) => setError(cause.message))
  }, [])
  return (
    <div className="page has-bg-image" style={{ backgroundImage: `url(${quizBackground})` }}>
      <PageHeader
        eyebrow="Speak your answer"
        title="My quizzes"
        description="Record answers in Sinhala, check the transcript and submit when you are ready."
      />
      {error && <Alert>{error}</Alert>}
      {!quizzes && !error ? (
        <Loading label="Loading quizzes…" />
      ) : quizzes?.length ? (
        <div className="quiz-grid">
          {quizzes.map((quiz) => (
            <article className="quiz-card" key={quiz.id}>
              <div className="quiz-card__top">
                <span className="document-icon document-icon--quiz_answer">
                  <Icon name="quiz" />
                </span>
                <StatusBadge status={quiz.submissionStatus} />
              </div>
              <h2>{quiz.title}</h2>
              <p>{quiz.description}</p>
              <div className="quiz-card__meta">
                <span>
                  <Icon name="file" size={16} /> {quiz.questions.length} questions
                </span>
                {quiz.dueDate && (
                  <span>
                    <Icon name="clock" size={16} /> Due{' '}
                    {new Intl.DateTimeFormat('en-GB', { day: 'numeric', month: 'short' }).format(
                      new Date(`${quiz.dueDate}T00:00:00`),
                    )}
                  </span>
                )}
              </div>
              <Link className="button button--primary" to={`/quizzes/${quiz.id}`}>
                {quiz.submissionStatus === 'SUBMITTED'
                  ? 'View submission'
                  : quiz.submissionStatus === 'IN_PROGRESS'
                    ? 'Continue quiz'
                    : 'Start quiz'}{' '}
                <Icon name="arrow" size={17} />
              </Link>
            </article>
          ))}
        </div>
      ) : (
        <EmptyState
          icon="quiz"
          title="No quizzes available"
          message="Published quizzes from your teachers will appear here."
        />
      )}
    </div>
  )
}

export function QuizAnswerPage() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [quiz, setQuiz] = useState(null)
  const [error, setError] = useState('')
  const [current, setCurrent] = useState(0)
  const [answers, setAnswers] = useState({})
  const [workingTranscript, setWorkingTranscript] = useState(null)
  const [confirm, setConfirm] = useState(false)
  const [submitted, setSubmitted] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const { job, start, reset } = useTranscriptionJob()
  const { interactionMode } = useAccessibility()
  const [commandFeedback, setCommandFeedback] = useState('')
  const showCommandFeedback = (message) => {
    setCommandFeedback(message)
    window.setTimeout(() => setCommandFeedback(''), 2200)
  }
  const voice = useVoiceCommands({
    onCommand: (command) => {
      if (!quiz || submitted) return
      if (command === 'next') {
        if (current < quiz.questions.length - 1 && Boolean(answers[quiz.questions[current].id])) {
          setCurrent((value) => value + 1)
          setWorkingTranscript(answers[quiz.questions[current + 1]?.id] || null)
          reset()
        } else {
          showCommandFeedback('Answer this question before moving on')
        }
      } else if (command === 'previous') {
        if (current > 0) {
          setCurrent((value) => value - 1)
          setWorkingTranscript(answers[quiz.questions[current - 1]?.id] || null)
          reset()
        } else {
          showCommandFeedback('Already on the first question')
        }
      } else if (command === 'submit') {
        const isLast = current === quiz.questions.length - 1
        const allDone = quiz.questions.filter((q) => q.required).every((q) => answers[q.id])
        if (isLast && allDone) {
          setConfirm(true)
        } else {
          showCommandFeedback('Complete every required question first')
        }
      }
    },
  })
  useEffect(() => {
    if (interactionMode !== 'command' && voice.isListening) voice.stop()
  }, [interactionMode]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    api
      .getQuiz(id)
      .then((item) => {
        setQuiz(item)
        if (item.submissionStatus === 'SUBMITTED') setSubmitted(true)
      })
      .catch((cause) => setError(cause.message))
  }, [id])
  useEffect(() => {
    if (job?.status === 'COMPLETED')
      api
        .getTranscript(job.transcriptId)
        .then(setWorkingTranscript)
        .catch((cause) => setError(cause.message))
  }, [job])
  if (error)
    return (
      <div className="page page--narrow has-bg-image" style={{ backgroundImage: `url(${quizBackground})` }}>
        <Alert>{error}</Alert>
      </div>
    )
  if (!quiz)
    return (
      <div className="page has-bg-image" style={{ backgroundImage: `url(${quizBackground})` }}>
        <Loading label="Opening quiz…" />
      </div>
    )
  if (submitted)
    return (
      <div className="page page--narrow has-bg-image" style={{ backgroundImage: `url(${quizBackground})` }}>
        <div className="success-state">
          <span>
            <Icon name="check" size={34} />
          </span>
          <span className="eyebrow">Submitted successfully</span>
          <h1>Your answers are on their way</h1>
          <p>
            Your teacher can now review the transcribed responses for <strong>{quiz.title}</strong>.
          </p>
          <button className="button button--primary" onClick={() => navigate('/quizzes')}>
            Back to my quizzes
          </button>
        </div>
      </div>
    )
  const question = quiz.questions[current]
  const answered = Boolean(answers[question.id])
  const completeCount = Object.keys(answers).length
  const allAnswered = quiz.questions.filter((q) => q.required).every((q) => answers[q.id])
  const record = (file) => {
    const data = new FormData()
    data.append('file', file)
    data.append('title', `${quiz.title} — Question ${current + 1}`)
    data.append('type', 'QUIZ_ANSWER')
    setWorkingTranscript(null)
    start(data)
  }
  const changed = (item) => {
    setWorkingTranscript(item)
    if (item.status === 'FINALIZED') setAnswers((value) => ({ ...value, [question.id]: item }))
  }
  const go = (next) => {
    setCurrent(next)
    setWorkingTranscript(answers[quiz.questions[next].id] || null)
    reset()
  }
  const submit = async () => {
    setConfirm(false)
    setSubmitting(true)
    try {
      await api.submitQuiz(
        quiz.id,
        Object.values(answers).map((item) => ({ transcriptId: item.id })),
      )
      setSubmitted(true)
    } catch (cause) {
      setError(cause.message)
    } finally {
      setSubmitting(false)
    }
  }
  return (
    <div className="page has-bg-image" style={{ backgroundImage: `url(${quizBackground})` }}>
      <PageHeader
        eyebrow={`Question ${current + 1} of ${quiz.questions.length}`}
        title={quiz.title}
        description={quiz.description}
        back={
          <button className="back-link" onClick={() => navigate('/quizzes')}>
            ← Exit quiz
          </button>
        }
        actions={
          <span className="question-count">
            <strong>{completeCount}</strong> / {quiz.questions.length} answered
          </span>
        }
      />
      <ProgressSteps steps={quiz.questions.map((_, index) => `Q${index + 1}`)} current={current} />
      <div className="quiz-workspace">
        <aside className="question-nav">
          <strong>Questions</strong>
          {quiz.questions.map((item, index) => (
            <button
              key={item.id}
              className={`${index === current ? 'active' : ''} ${answers[item.id] ? 'answered' : ''}`}
              onClick={() => go(index)}
            >
              <span>{answers[item.id] ? '✓' : index + 1}</span>
              <em>Question {index + 1}</em>
            </button>
          ))}
        </aside>
        <section className="answer-panel">
          <div className="question-prompt">
            <span>
              Question {current + 1}
              {question.required && <em>Required</em>}
            </span>
            <h2 lang="si">{question.text}</h2>
          </div>
          {answered && !workingTranscript && (
            <Alert type="success" title="Answer ready">
              You have completed this question.
            </Alert>
          )}
          {job && !workingTranscript ? (
            <TranscriptionStatus
              status={job.status}
              message={job.message}
              onRetry={() => reset()}
            />
          ) : workingTranscript ? (
            <TranscriptEditor
              initialTranscript={workingTranscript}
              compact
              onTranscriptChange={changed}
            />
          ) : (
            <>
              <h3>Record your spoken answer</h3>
              <AudioRecorder onUse={record} />
              <p className="privacy-inline">
                <Icon name="help" size={15} /> Your recording is stored for transcription and
                teacher review.
              </p>
            </>
          )}
          {interactionMode === 'command' && voice.error && <Alert>{voice.error}</Alert>}
          {interactionMode === 'command' && commandFeedback && (
            <p className="command-feedback command-feedback--advisory" role="status">
              <Icon name="alert" size={14} /> {commandFeedback}
            </p>
          )}
          <div className="quiz-navigation">
            {interactionMode === 'command' && (
              <div className="voice-command-toggle">
                <button
                  type="button"
                  className={`button button--icon ${voice.isListening ? 'is-recording' : ''}`}
                  onClick={voice.isListening ? voice.stop : voice.start}
                  disabled={voice.status === 'connecting' || voice.status === 'stopping'}
                  aria-label={voice.isListening ? 'Stop voice commands' : 'Listen for voice commands'}
                  title={voice.isListening ? 'Stop voice commands' : 'Say "next", "previous" or "submit"'}
                >
                  <Icon name={voice.isListening ? 'stop' : 'mic'} size={17} />
                </button>
                {voice.isListening && (
                  <VoiceMeter registerBar={voice.registerBar} active={voice.voiceDetected} compact />
                )}
              </div>
            )}
            <button
              className="button button--secondary"
              disabled={current === 0}
              onClick={() => go(current - 1)}
            >
              Previous
            </button>
            {current < quiz.questions.length - 1 ? (
              <button
                className="button button--primary"
                disabled={!answered}
                onClick={() => go(current + 1)}
              >
                Next question <Icon name="arrow" size={17} />
              </button>
            ) : (
              <button
                className="button button--primary"
                disabled={!allAnswered || submitting}
                onClick={() => setConfirm(true)}
              >
                {submitting ? 'Submitting…' : 'Review & submit'}
              </button>
            )}
          </div>
          {!allAnswered && current === quiz.questions.length - 1 && (
            <p className="field-hint">Complete every required question before final submission.</p>
          )}
        </section>
      </div>
      <ConfirmDialog
        open={confirm}
        title="Submit your quiz?"
        message="Your answers will be sent to your teacher. You will not be able to edit them after submission."
        confirmLabel="Submit quiz"
        onCancel={() => setConfirm(false)}
        onConfirm={submit}
      />
    </div>
  )
}
