import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api } from '../api'
import submissionsBackground from '../assets/5.png'
import Icon from '../components/Icon'
import { Alert, EmptyState, Loading, PageHeader, StatusBadge } from '../components/UI'
import { useLanguage } from '../contexts/LanguageContext'
import { useToast } from '../contexts/ToastContext'

export function SubmissionsPage() {
  const { t, language } = useLanguage()
  const [items, setItems] = useState(null)
  const [error, setError] = useState('')
  useEffect(() => {
    api
      .getSubmissions()
      .then(setItems)
      .catch((cause) => setError(cause.message))
  }, [])
  return (
    <div className="page has-bg-image" style={{ backgroundImage: `url(${submissionsBackground})` }}>
      <PageHeader
        eyebrow={t('teacherQuiz.workspace')}
        title={t('nav.reviewSubmissions')}
        description={t('dashboard.teacher.submissionsDescription')}
      />
      {error && <Alert>{error}</Alert>}
      {!items && !error ? (
        <Loading label={t('submissions.loading')} />
      ) : items?.length ? (
        <div className="table-card">
          <table>
            <thead>
              <tr>
                <th>{t('submissions.colStudent')}</th>
                <th>{t('submissions.colQuiz')}</th>
                <th>{t('submissions.colSubmitted')}</th>
                <th>{t('submissions.colStatus')}</th>
                <th>
                  <span className="sr-only">{t('submissions.colAction')}</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {items.map((item) => (
                <tr key={item.id}>
                  <td>
                    <div className="student-cell">
                      <span className="avatar avatar--small">
                        {item.studentName
                          .split(' ')
                          .map((part) => part[0])
                          .join('')}
                      </span>
                      <strong>{item.studentName}</strong>
                    </div>
                  </td>
                  <td>{item.quizTitle}</td>
                  <td>
                    {new Intl.DateTimeFormat(language === 'si' ? 'si-LK' : 'en-GB', {
                      day: 'numeric',
                      month: 'short',
                      year: 'numeric',
                      hour: '2-digit',
                      minute: '2-digit',
                    }).format(new Date(item.submittedAt))}
                  </td>
                  <td>
                    <StatusBadge status={item.status} />
                  </td>
                  <td>
                    <Link
                      className="button button--secondary button--small"
                      to={`/teacher/submissions/${item.id}`}
                    >
                      {t('submissions.openReview')} <Icon name="arrow" size={15} />
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <EmptyState
          icon="users"
          title={t('submissions.noSubmissionsYet')}
          message={t('submissions.noSubmissionsMessage')}
        />
      )}
    </div>
  )
}

export function SubmissionReviewPage() {
  const { t, language } = useLanguage()
  const { id } = useParams()
  const navigate = useNavigate()
  const { showToast } = useToast()
  const [item, setItem] = useState(null)
  const [review, setReview] = useState({ mark: '', feedback: '' })
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  useEffect(() => {
    api
      .getSubmission(id)
      .then((result) => {
        setItem(result)
        setReview({ mark: result.mark || '', feedback: result.feedback || '' })
      })
      .catch((cause) => setError(cause.message))
  }, [id])
  const save = async () => {
    if (review.mark !== '' && (Number(review.mark) < 0 || Number(review.mark) > 100)) {
      setError(t('submissions.markRange'))
      return
    }
    setSaving(true)
    setError('')
    try {
      const result = await api.reviewSubmission(id, review)
      setItem(result)
      showToast(t('submissions.reviewSaved'))
    } catch (cause) {
      setError(cause.message)
    } finally {
      setSaving(false)
    }
  }
  if (error && !item)
    return (
      <div className="page page--narrow has-bg-image" style={{ backgroundImage: `url(${submissionsBackground})` }}>
        <Alert>{error}</Alert>
      </div>
    )
  if (!item)
    return (
      <div className="page has-bg-image" style={{ backgroundImage: `url(${submissionsBackground})` }}>
        <Loading label={t('submissions.openingSubmission')} />
      </div>
    )
  return (
    <div className="page page--narrow has-bg-image" style={{ backgroundImage: `url(${submissionsBackground})` }}>
      <PageHeader
        eyebrow={t('submissions.submissionReview')}
        title={item.studentName}
        description={t(
          'submissions.quizSubmittedOn',
          item.quizTitle,
          new Intl.DateTimeFormat(language === 'si' ? 'si-LK' : 'en-GB', {
            day: 'numeric',
            month: 'long',
            year: 'numeric',
          }).format(new Date(item.submittedAt)),
        )}
        actions={<StatusBadge status={item.status} />}
        back={
          <button className="back-link" onClick={() => navigate('/teacher/submissions')}>
            {t('submissions.backToSubmissions')}
          </button>
        }
      />
      {error && <Alert>{error}</Alert>}
      <section className="answers-review">
        <h2>{t('submissions.transcribedAnswers')}</h2>
        {item.answers.map((answer, index) => (
          <article key={answer.questionId}>
            <span>{t('submissions.questionN', index + 1)}</span>
            <h3 lang="si">{answer.question}</h3>
            <div className="answer-transcript">
              <Icon name="file" size={18} />
              <p lang="si">{answer.transcript}</p>
            </div>
          </article>
        ))}
      </section>
      <section className="form-card">
        <h2>{t('submissions.markFeedback')}</h2>
        <div className="field field--mark">
          <label htmlFor="mark">
            {t('submissions.mark')} <span>{t('submissions.outOf100')}</span>
          </label>
          <div>
            <input
              id="mark"
              type="number"
              min="0"
              max="100"
              value={review.mark}
              onChange={(e) => setReview({ ...review, mark: e.target.value })}
            />
            <span>/ 100</span>
          </div>
        </div>
        <div className="field">
          <label htmlFor="feedback">{t('submissions.feedback')}</label>
          <textarea
            id="feedback"
            rows="5"
            value={review.feedback}
            onChange={(e) => setReview({ ...review, feedback: e.target.value })}
            placeholder={t('submissions.feedbackPlaceholder')}
          />
        </div>
        <div className="button-row button-row--end">
          <button className="button button--primary" onClick={save} disabled={saving}>
            {saving ? (
              <>
                <span className="spinner spinner--small" /> {t('submissions.saving')}
              </>
            ) : (
              <>
                <Icon name="check" size={17} /> {t('submissions.saveReview')}
              </>
            )}
          </button>
        </div>
      </section>
    </div>
  )
}
