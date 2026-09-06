import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api } from '../api'
import quizBackground from '../assets/4.jpg'
import AudioRecorder from '../components/AudioRecorder'
import Icon from '../components/Icon'
import LiveTranscription from '../components/LiveTranscription'
import TranscriptEditor from '../components/TranscriptEditor'
import TranscriptionStatus from '../components/TranscriptionStatus'
import useTranscriptionJob from '../components/useTranscriptionJob'
import useVoiceCommands from '../hooks/useVoiceCommands'
import VoiceMeter from '../components/VoiceMeter'
import { useAccessibility } from '../contexts/AccessibilityContext'
import { useLanguage } from '../contexts/LanguageContext'
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
  const { t, language } = useLanguage()
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
        eyebrow={t('quiz.speakYourAnswer')}
        title={t('quiz.myQuizzes')}
        description={t('quiz.listDescription')}
      />
      {error && <Alert>{error}</Alert>}
      {!quizzes && !error ? (
        <Loading label={t('quiz.loadingQuizzes')} />
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
                  <Icon name="file" size={16} /> {t('quiz.questionsCount', quiz.questions.length)}
                </span>
                {quiz.dueDate && (
                  <span>
                    <Icon name="clock" size={16} />{' '}
                    {t(
                      'quiz.due',
                      new Intl.DateTimeFormat(language === 'si' ? 'si-LK' : 'en-GB', {
                        day: 'numeric',
                        month: 'short',
                      }).format(new Date(`${quiz.dueDate}T00:00:00`)),
                    )}
                  </span>
                )}
              </div>
              <Link className="button button--primary" to={`/quizzes/${quiz.id}`}>
                {quiz.submissionStatus === 'REVIEWED'
                  ? t('quiz.viewResults')
                  : quiz.submissionStatus === 'SUBMITTED'
                    ? t('quiz.viewSubmission')
                    : quiz.submissionStatus === 'IN_PROGRESS'
                      ? t('quiz.continueQuiz')
                      : t('quiz.startQuiz')}{' '}
                <Icon name="arrow" size={17} />
              </Link>
            </article>
          ))}
        </div>
      ) : (
        <EmptyState icon="quiz" title={t('quiz.noQuizzesAvailable')} message={t('quiz.noQuizzesMessage')} />
      )}
    </div>
  )
}

export function QuizAnswerPage() {
  const { t } = useLanguage()
  const { id } = useParams()
  const navigate = useNavigate()
  const [quiz, setQuiz] = useState(null)
  const [error, setError] = useState('')
  const [current, setCurrent] = useState(0)
  const [answers, setAnswers] = useState({})
  const [workingTranscript, setWorkingTranscript] = useState(null)
  const [liveLoading, setLiveLoading] = useState(false)
  const [answering, setAnswering] = useState(false)
  const [confirm, setConfirm] = useState(false)
  const [submitted, setSubmitted] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  // Only populated once the teacher has reviewed the submission - the
  // mark/feedback/per-answer correctness this holds is what turns the
  // plain "your answers are on their way" screen below into the full
  // results view.
  const [mySubmission, setMySubmission] = useState(null)
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
      // The "submit your quiz?" dialog takes priority over every other
      // command while it's open: saying "submit" again confirms it,
      // "cancel" dismisses it - neither should fall through to the
      // per-question handling below.
      if (confirm) {
        if (command === 'submit') {
          submit()
        } else if (command === 'cancel') {
          setConfirm(false)
          showCommandFeedback(t('quiz.submitCancelled'))
        }
        return
      }
      if (command === 'next') {
        if (current < quiz.questions.length - 1 && Boolean(answers[quiz.questions[current].id])) {
          setCurrent((value) => value + 1)
          setWorkingTranscript(answers[quiz.questions[current + 1]?.id] || null)
          setAnswering(false)
          reset()
        } else {
          showCommandFeedback(t('quiz.answerBeforeMoving'))
        }
      } else if (command === 'previous') {
        if (current > 0) {
          setCurrent((value) => value - 1)
          setWorkingTranscript(answers[quiz.questions[current - 1]?.id] || null)
          setAnswering(false)
          reset()
        } else {
          showCommandFeedback(t('quiz.alreadyFirstQuestion'))
        }
      } else if (command === 'submit') {
        const isLast = current === quiz.questions.length - 1
        const allDone = quiz.questions.filter((q) => q.required).every((q) => answers[q.id])
        if (isLast && allDone) {
          setConfirm(true)
        } else {
          showCommandFeedback(t('quiz.completeAllRequired'))
        }
      } else if (['option_1', 'option_2', 'option_3', 'option_4'].includes(command)) {
        const question = quiz.questions[current]
        if (question.type !== 'MCQ') return
        const index = Number(command.slice(-1)) - 1
        const option = question.options[index]
        if (!option) {
          showCommandFeedback(t('quiz.mcqOptionUnavailable', index + 1))
          return
        }
        setAnswers((value) => ({
          ...value,
          [question.id]: { selectedOptionId: option.id },
        }))
        showCommandFeedback(t('quiz.mcqOptionSelected', index + 1))
      } else if (command === 'cancel') {
        const question = quiz.questions[current]
        if (question.type !== 'MCQ') return
        setAnswers((value) => {
          const next = { ...value }
          delete next[question.id]
          return next
        })
        showCommandFeedback(t('quiz.mcqSelectionCleared'))
      } else if (command === 'answer') {
        // Hands off from the always-on command mic to the live
        // transcription mic for a written-answer question - only
        // meaningful there, and only when nothing is already in
        // progress for this question.
        const question = quiz.questions[current]
        if (question.type === 'MCQ' || job || workingTranscript || answering) return
        setAnswering(true)
        voice.stop()
      } else if (command === 'save') {
        if (!workingTranscript) {
          showCommandFeedback(t('editor.nothingToSave'))
          return
        }
        setAnswers((value) => ({ ...value, [quiz.questions[current].id]: workingTranscript }))
        showCommandFeedback(t('quiz.answerSaved'))
      }
    },
  })
  useEffect(() => {
    if (interactionMode !== 'command' && voice.isListening) voice.stop()
  }, [interactionMode]) // eslint-disable-line react-hooks/exhaustive-deps
  // The command mic stays on for the whole quiz in command-interaction
  // mode - it's only ever paused while "answering" (the live
  // transcription mic has taken over) or once the quiz is submitted.
  useEffect(() => {
    if (
      interactionMode === 'command' &&
      quiz &&
      !submitted &&
      !answering &&
      voice.status === 'idle'
    ) {
      voice.start()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [interactionMode, quiz, submitted, answering])
  useEffect(() => {
    api
      .getQuiz(id)
      .then((item) => {
        setQuiz(item)
        if (item.submissionStatus === 'SUBMITTED' || item.submissionStatus === 'REVIEWED') {
          setSubmitted(true)
        }
        if (item.submissionStatus === 'REVIEWED' && item.submissionId) {
          api
            .getSubmission(item.submissionId)
            .then(setMySubmission)
            .catch((cause) => setError(cause.message))
        }
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
        <Loading label={t('quiz.openingQuiz')} />
      </div>
    )
  if (submitted && quiz.submissionStatus === 'REVIEWED') {
    if (!mySubmission)
      return (
        <div className="page has-bg-image" style={{ backgroundImage: `url(${quizBackground})` }}>
          <Loading label={t('quiz.openingResults')} />
        </div>
      )
    return (
      <div className="page page--narrow has-bg-image" style={{ backgroundImage: `url(${quizBackground})` }}>
        <PageHeader
          eyebrow={t('quiz.results')}
          title={quiz.title}
          description={t('quiz.reviewedBy')}
          back={
            <button className="back-link" onClick={() => navigate('/quizzes')}>
              {t('quiz.backToMyQuizzes')}
            </button>
          }
        />
        {error && <Alert>{error}</Alert>}
        <section className="form-card">
          <h2>{t('quiz.yourMark')}</h2>
          <p className="quiz-results__mark">
            {mySubmission.mark !== null && mySubmission.mark !== undefined
              ? `${mySubmission.mark} / 100`
              : t('quiz.notMarkedYet')}
          </p>
          <h2>{t('quiz.teacherFeedback')}</h2>
          <p>{mySubmission.feedback || t('quiz.noFeedbackGiven')}</p>
        </section>
        <section className="answers-review">
          <h2>{t('quiz.yourAnswers')}</h2>
          {mySubmission.answers.map((answer, index) => (
            <article key={answer.questionId}>
              <span>{t('submissions.questionN', index + 1)}</span>
              <h3 lang="si">{answer.question}</h3>
              {answer.type === 'MCQ' ? (
                <>
                  <div className="mcq-review-options">
                    {(answer.options || []).map((option, optIndex) => {
                      const isSelected = option.id === answer.selectedOptionId
                      const classes = ['mcq-review-option']
                      if (isSelected) classes.push('mcq-review-option--selected')
                      if (option.isCorrect) classes.push('mcq-review-option--correct')
                      if (isSelected && !option.isCorrect)
                        classes.push('mcq-review-option--incorrect-selection')
                      return (
                        <div className={classes.join(' ')} key={option.id}>
                          <span>{optIndex + 1}.</span>
                          <span lang="si">{option.text}</span>
                          {isSelected && <em>{t('submissions.mcqStudentSelected')}</em>}
                          {option.isCorrect && <strong>{t('submissions.mcqCorrectAnswer')}</strong>}
                        </div>
                      )
                    })}
                  </div>
                  <p
                    className={`mcq-review-result ${
                      answer.isCorrect ? 'mcq-review-result--correct' : 'mcq-review-result--incorrect'
                    }`}
                  >
                    <Icon name={answer.isCorrect ? 'check' : 'alert'} size={15} />{' '}
                    {answer.isCorrect ? t('submissions.mcqCorrect') : t('submissions.mcqIncorrect')}
                  </p>
                </>
              ) : (
                <div className="answer-transcript">
                  <Icon name="file" size={18} />
                  <p lang="si">{answer.transcript}</p>
                </div>
              )}
            </article>
          ))}
        </section>
      </div>
    )
  }
  if (submitted)
    return (
      <div className="page page--narrow has-bg-image" style={{ backgroundImage: `url(${quizBackground})` }}>
        <div className="success-state">
          <span>
            <Icon name="check" size={34} />
          </span>
          <span className="eyebrow">{t('quiz.submittedSuccessfully')}</span>
          <h1>{t('quiz.answersOnWay')}</h1>
          <p>
            {t('quiz.teacherCanReviewPrefix')}
            <strong>{quiz.title}</strong>
            {t('quiz.teacherCanReviewSuffix')}
          </p>
          <button className="button button--primary" onClick={() => navigate('/quizzes')}>
            {t('quiz.backToMyQuizzes')}
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
  const handleLiveSessionEnd = (transcriptId) => {
    setLiveLoading(true)
    setAnswering(false)
    api
      .getTranscript(transcriptId)
      .then(setWorkingTranscript)
      .catch((cause) => setError(cause.message))
      .finally(() => setLiveLoading(false))
  }
  const changed = (item) => {
    setWorkingTranscript(item)
    // Saving the spoken-answer transcript (no separate finalize step here)
    // is what marks this question answered - the quiz itself is only
    // actually sent to the teacher at the final "Submit quiz" step.
    setAnswers((value) => ({ ...value, [question.id]: item }))
  }
  const selectOption = (option) => {
    setAnswers((value) => ({
      ...value,
      [question.id]: { selectedOptionId: option.id },
    }))
  }
  const go = (next) => {
    setCurrent(next)
    setWorkingTranscript(answers[quiz.questions[next].id] || null)
    setAnswering(false)
    reset()
  }
  const submit = async () => {
    setConfirm(false)
    setSubmitting(true)
    try {
      await api.submitQuiz(
        quiz.id,
        quiz.questions
          .filter((q) => answers[q.id])
          .map((q) =>
            q.type === 'MCQ'
              ? { questionId: q.id, selectedOptionId: answers[q.id].selectedOptionId }
              : { questionId: q.id, transcriptId: answers[q.id].id },
          ),
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
        eyebrow={t('quiz.questionOf', current + 1, quiz.questions.length)}
        title={quiz.title}
        description={quiz.description}
        back={
          <button className="back-link" onClick={() => navigate('/quizzes')}>
            {t('quiz.exitQuiz')}
          </button>
        }
        actions={
          <span className="question-count">
            <strong>{completeCount}</strong> / {quiz.questions.length} {t('quiz.answeredCount')}
          </span>
        }
      />
      <ProgressSteps steps={quiz.questions.map((_, index) => `Q${index + 1}`)} current={current} />
      <div className="quiz-workspace">
        <aside className="question-nav">
          <strong>{t('quiz.questions')}</strong>
          {quiz.questions.map((item, index) => (
            <button
              key={item.id}
              className={`${index === current ? 'active' : ''} ${answers[item.id] ? 'answered' : ''}`}
              onClick={() => go(index)}
            >
              <span>{answers[item.id] ? '✓' : index + 1}</span>
              <em>{t('quiz.questionN', index + 1)}</em>
            </button>
          ))}
        </aside>
        <section className="answer-panel">
          <div className="question-prompt">
            <span>
              {t('quiz.questionN', current + 1)}
              {question.required && <em>{t('quiz.required')}</em>}
            </span>
            <h2 lang="si">{question.text}</h2>
          </div>
          {question.type === 'MCQ' ? (
            <div className="mcq-options">
              <p className="field-hint">
                {t('quiz.mcqHint')}
                {interactionMode === 'command' && ` ${t('quiz.sayOneToFourOrCancel')}.`}
              </p>
              {question.options.map((option, index) => {
                const isSelected = answers[question.id]?.selectedOptionId === option.id
                return (
                  <button
                    key={option.id}
                    type="button"
                    className={`mcq-option ${isSelected ? 'mcq-option--selected' : ''}`}
                    onClick={() => selectOption(option)}
                  >
                    <span className="mcq-option__number">{index + 1}</span>
                    <span lang="si">{option.text}</span>
                    {isSelected && (
                      <span className="mcq-option__badge">
                        <Icon name="check" size={14} /> {t('quiz.mcqSelected')}
                      </span>
                    )}
                  </button>
                )
              })}
            </div>
          ) : (
            <>
              <p className="field-hint">{t('quiz.spokenHint')}</p>
              {answered && !workingTranscript && (
                <Alert type="success" title={t('quiz.answerReady')}>
                  {t('quiz.questionCompleted')}
                </Alert>
              )}
              {job && !workingTranscript ? (
                <TranscriptionStatus
                  status={job.status}
                  message={job.message}
                  onRetry={() => reset()}
                />
              ) : liveLoading ? (
                <Loading label={t('quiz.preparingAnswer')} />
              ) : workingTranscript ? (
                <TranscriptEditor
                  initialTranscript={workingTranscript}
                  compact
                  onTranscriptChange={changed}
                />
              ) : (
                <>
                  <h3>{t('quiz.liveTranscription')}</h3>
                  {interactionMode === 'command' && (
                    <p className="field-hint">{t('quiz.sayAnswerToDictate')}</p>
                  )}
                  <LiveTranscription
                    title={`${quiz.title} — Question ${current + 1}`}
                    mode="EXAM"
                    onSessionEnd={handleLiveSessionEnd}
                    autoStart={answering}
                  />
                  <p className="recorder__divider">{t('quiz.orRecord')}</p>
                  <h3>{t('quiz.recordSpokenAnswer')}</h3>
                  <AudioRecorder onUse={record} showUpload={false} />
                  <p className="privacy-inline">
                    <Icon name="help" size={15} /> {t('quiz.recordingStoredNotice')}
                  </p>
                </>
              )}
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
                  aria-label={voice.isListening ? t('quiz.stopVoiceCommands') : t('quiz.listenVoiceCommands')}
                  title={
                    voice.isListening
                      ? t('quiz.stopVoiceCommands')
                      : question.type === 'MCQ'
                        ? t('quiz.sayOneToFourOrCancel')
                        : t('quiz.sayAnswerToDictate')
                  }
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
              {t('quiz.previous')}
            </button>
            {current < quiz.questions.length - 1 ? (
              <button
                className="button button--primary"
                disabled={!answered}
                onClick={() => go(current + 1)}
              >
                {t('quiz.nextQuestion')} <Icon name="arrow" size={17} />
              </button>
            ) : (
              <button
                className="button button--primary"
                disabled={!allAnswered || submitting}
                onClick={() => setConfirm(true)}
              >
                {submitting ? t('quiz.submitting') : t('quiz.reviewSubmit')}
              </button>
            )}
          </div>
          {!allAnswered && current === quiz.questions.length - 1 && (
            <p className="field-hint">{t('quiz.completeBeforeFinal')}</p>
          )}
        </section>
      </div>
      <ConfirmDialog
        open={confirm}
        title={t('quiz.submitYourQuiz')}
        message={t('quiz.submitMessage')}
        confirmLabel={t('quiz.submitQuiz')}
        onCancel={() => setConfirm(false)}
        onConfirm={submit}
      />
    </div>
  )
}
