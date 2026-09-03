import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api } from '../api'
import teacherQuizBackground from '../assets/workspace-background.jpg'
import Icon from '../components/Icon'
import {
  Alert,
  ConfirmDialog,
  EmptyState,
  Loading,
  PageHeader,
  StatusBadge,
} from '../components/UI'
import { useToast } from '../contexts/ToastContext'

export function TeacherQuizListPage() {
  const [items, setItems] = useState(null)
  const [filter, setFilter] = useState('ALL')
  const [error, setError] = useState('')
  useEffect(() => {
    api
      .getQuizzes({ teacher: true })
      .then(setItems)
      .catch((cause) => setError(cause.message))
  }, [])
  const shown = (items || []).filter((item) => filter === 'ALL' || item.status === filter)
  return (
    <div className="page has-bg-image" style={{ backgroundImage: `url(${teacherQuizBackground})` }}>
      <PageHeader
        eyebrow="Teacher workspace"
        title="Manage quizzes"
        description="Create speech-based quizzes and publish them when they are ready."
        actions={
          <Link className="button button--primary" to="/teacher/quizzes/new">
            <Icon name="plus" size={17} /> Create quiz
          </Link>
        }
      />
      <div className="filter-tabs" role="tablist" aria-label="Quiz status filter">
        {['ALL', 'DRAFT', 'PUBLISHED'].map((item) => (
          <button
            key={item}
            role="tab"
            aria-selected={filter === item}
            className={filter === item ? 'active' : ''}
            onClick={() => setFilter(item)}
          >
            {item === 'ALL' ? 'All quizzes' : item[0] + item.slice(1).toLowerCase()}
          </button>
        ))}
      </div>
      {error && <Alert>{error}</Alert>}
      {!items && !error ? (
        <Loading label="Loading quizzes…" />
      ) : shown.length ? (
        <div className="manage-list">
          {shown.map((quiz) => (
            <article key={quiz.id}>
              <div>
                <span className="document-icon document-icon--quiz_answer">
                  <Icon name="quiz" />
                </span>
                <span>
                  <strong>{quiz.title}</strong>
                  <small>
                    {quiz.questions.length} questions{' '}
                    {quiz.dueDate ? `· Due ${quiz.dueDate}` : '· No due date'}
                  </small>
                </span>
              </div>
              <StatusBadge status={quiz.status} />
              <Link
                className="button button--secondary button--small"
                to={`/teacher/quizzes/${quiz.id}/edit`}
              >
                <Icon name="edit" size={15} /> Edit
              </Link>
            </article>
          ))}
        </div>
      ) : (
        <EmptyState
          icon="quiz"
          title="No quizzes here"
          message="Create a quiz or change the selected status filter."
        />
      )}
    </div>
  )
}

export function QuizFormPage() {
  const { id } = useParams()
  const navigate = useNavigate()
  const { showToast } = useToast()
  const [loading, setLoading] = useState(Boolean(id))
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [removeIndex, setRemoveIndex] = useState(null)
  const [publishConfirm, setPublishConfirm] = useState(false)
  const [quiz, setQuiz] = useState({
    title: '',
    description: '',
    dueDate: '',
    status: 'DRAFT',
    questions: [{ id: `q-${Date.now()}`, text: '', required: true }],
  })
  useEffect(() => {
    if (id)
      api
        .getQuiz(id)
        .then((item) => {
          setQuiz(item)
          setLoading(false)
        })
        .catch((cause) => {
          setError(cause.message)
          setLoading(false)
        })
  }, [id])
  const validate = () => {
    if (!quiz.title.trim()) return 'Quiz title is required.'
    if (!quiz.questions.length) return 'Add at least one question.'
    if (quiz.questions.some((q) => !q.text.trim())) return 'Every question needs text.'
    return ''
  }
  const save = async (publish = false) => {
    const issue = validate()
    if (issue) {
      setError(issue)
      return
    }
    setSaving(true)
    setError('')
    try {
      let result = await api.saveQuiz({ ...quiz, status: publish ? quiz.status : 'DRAFT' })
      if (publish) result = await api.publishQuiz(result.id)
      setQuiz(result)
      showToast(publish ? 'Quiz published to students.' : 'Quiz saved as draft.')
      navigate('/teacher/quizzes')
    } catch (cause) {
      setError(cause.message)
    } finally {
      setSaving(false)
    }
  }
  const updateQuestion = (index, text) =>
    setQuiz((value) => ({
      ...value,
      questions: value.questions.map((q, i) => (i === index ? { ...q, text } : q)),
    }))
  const move = (index, direction) => {
    const questions = [...quiz.questions]
    const [item] = questions.splice(index, 1)
    questions.splice(index + direction, 0, item)
    setQuiz({ ...quiz, questions })
  }
  if (loading)
    return (
      <div className="page has-bg-image" style={{ backgroundImage: `url(${teacherQuizBackground})` }}>
        <Loading label="Loading quiz editor…" />
      </div>
    )
  return (
    <div className="page page--narrow has-bg-image" style={{ backgroundImage: `url(${teacherQuizBackground})` }}>
      <PageHeader
        eyebrow={id ? 'Edit quiz' : 'New quiz'}
        title={id ? quiz.title : 'Create a speech quiz'}
        description="Add simple text questions that students will answer using recorded speech."
        back={
          <button className="back-link" onClick={() => navigate('/teacher/quizzes')}>
            ← Back to quizzes
          </button>
        }
      />
      {error && <Alert>{error}</Alert>}
      <section className="form-card">
        <h2>Quiz details</h2>
        <div className="field">
          <label htmlFor="quiz-title">Title</label>
          <input
            id="quiz-title"
            value={quiz.title}
            onChange={(e) => setQuiz({ ...quiz, title: e.target.value })}
            placeholder="e.g. Data Structures — Week 3"
          />
        </div>
        <div className="field">
          <label htmlFor="quiz-description">Description</label>
          <textarea
            id="quiz-description"
            rows="3"
            value={quiz.description}
            onChange={(e) => setQuiz({ ...quiz, description: e.target.value })}
            placeholder="Tell students what to expect."
          />
        </div>
        <div className="field field--half">
          <label htmlFor="quiz-due">
            Due date <span>Optional</span>
          </label>
          <input
            id="quiz-due"
            type="date"
            value={quiz.dueDate}
            onChange={(e) => setQuiz({ ...quiz, dueDate: e.target.value })}
          />
        </div>
      </section>
      <section className="form-card">
        <div className="section-heading">
          <div>
            <h2>Questions</h2>
            <p>Students will record one spoken answer for each question.</p>
          </div>
          <button
            className="button button--secondary button--small"
            onClick={() =>
              setQuiz({
                ...quiz,
                questions: [...quiz.questions, { id: `q-${Date.now()}`, text: '', required: true }],
              })
            }
          >
            <Icon name="plus" size={16} /> Add question
          </button>
        </div>
        <div className="question-editor-list">
          {quiz.questions.map((question, index) => (
            <div className="question-editor" key={question.id}>
              <span className="drag-number">{index + 1}</span>
              <div className="field">
                <label htmlFor={`q-text-${question.id}`}>Question {index + 1}</label>
                <textarea
                  id={`q-text-${question.id}`}
                  lang="si"
                  rows="2"
                  value={question.text}
                  onChange={(e) => updateQuestion(index, e.target.value)}
                  placeholder="Enter the question text…"
                />
              </div>
              <div className="question-editor__actions">
                <button
                  className="icon-button"
                  aria-label="Move question up"
                  disabled={index === 0}
                  onClick={() => move(index, -1)}
                >
                  ↑
                </button>
                <button
                  className="icon-button"
                  aria-label="Move question down"
                  disabled={index === quiz.questions.length - 1}
                  onClick={() => move(index, 1)}
                >
                  ↓
                </button>
                <button
                  className="icon-button icon-button--danger"
                  aria-label="Remove question"
                  onClick={() => setRemoveIndex(index)}
                >
                  <Icon name="trash" size={17} />
                </button>
              </div>
            </div>
          ))}
        </div>
      </section>
      <div className="sticky-actions">
        <button className="button button--secondary" onClick={() => save(false)} disabled={saving}>
          Save as draft
        </button>
        <button
          className="button button--primary"
          onClick={() => {
            const issue = validate()
            if (issue) setError(issue)
            else setPublishConfirm(true)
          }}
          disabled={saving}
        >
          {saving ? 'Saving…' : 'Publish quiz'}
        </button>
      </div>
      <ConfirmDialog
        open={removeIndex !== null}
        dangerous
        title="Remove this question?"
        message="The question will be removed from this quiz. This cannot be undone after saving."
        confirmLabel="Remove question"
        onCancel={() => setRemoveIndex(null)}
        onConfirm={() => {
          setQuiz({
            ...quiz,
            questions: quiz.questions.filter((_, index) => index !== removeIndex),
          })
          setRemoveIndex(null)
        }}
      />
      <ConfirmDialog
        open={publishConfirm}
        title="Publish this quiz?"
        message="Students will be able to see and answer it immediately."
        confirmLabel="Publish quiz"
        onCancel={() => setPublishConfirm(false)}
        onConfirm={() => {
          setPublishConfirm(false)
          save(true)
        }}
      />
    </div>
  )
}
