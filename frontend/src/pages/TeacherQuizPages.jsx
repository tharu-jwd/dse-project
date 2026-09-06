import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api } from '../api'
import teacherQuizBackground from '../assets/6.jpg'
import Icon from '../components/Icon'
import {
  Alert,
  ConfirmDialog,
  EmptyState,
  Loading,
  PageHeader,
  StatusBadge,
} from '../components/UI'
import { useLanguage } from '../contexts/LanguageContext'
import { useToast } from '../contexts/ToastContext'

export function TeacherQuizListPage() {
  const { t } = useLanguage()
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
        eyebrow={t('teacherQuiz.workspace')}
        title={t('teacherQuiz.manageQuizzes')}
        description={t('teacherQuiz.listDescription')}
        actions={
          <Link className="button button--primary" to="/teacher/quizzes/new">
            <Icon name="plus" size={17} /> {t('teacherQuiz.createQuiz')}
          </Link>
        }
      />
      <div className="filter-tabs" role="tablist" aria-label={t('teacherQuiz.statusFilter')}>
        {['ALL', 'DRAFT', 'PUBLISHED'].map((item) => (
          <button
            key={item}
            role="tab"
            aria-selected={filter === item}
            className={filter === item ? 'active' : ''}
            onClick={() => setFilter(item)}
          >
            {item === 'ALL'
              ? t('teacherQuiz.allQuizzes')
              : item === 'DRAFT'
                ? t('teacherQuiz.statusDraft')
                : t('teacherQuiz.statusPublished')}
          </button>
        ))}
      </div>
      {error && <Alert>{error}</Alert>}
      {!items && !error ? (
        <Loading label={t('teacherQuiz.loadingQuizzes')} />
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
                    {t('teacherQuiz.questionsCount', quiz.questions.length)}{' '}
                    {quiz.dueDate ? t('teacherQuiz.dueDateLabel', quiz.dueDate) : t('teacherQuiz.noDueDate')}
                  </small>
                </span>
              </div>
              <StatusBadge status={quiz.status} />
              <Link
                className="button button--secondary button--small"
                to={`/teacher/quizzes/${quiz.id}/edit`}
              >
                <Icon name="edit" size={15} /> {t('teacherQuiz.edit')}
              </Link>
            </article>
          ))}
        </div>
      ) : (
        <EmptyState
          icon="quiz"
          title={t('teacherQuiz.noQuizzesHere')}
          message={t('teacherQuiz.noQuizzesMessage')}
        />
      )}
    </div>
  )
}

export function QuizFormPage() {
  const { t } = useLanguage()
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
    questions: [
      {
        id: `q-${Date.now()}`,
        text: '',
        type: 'SPOKEN',
        required: true,
        options: [],
      },
    ],
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
    if (!quiz.title.trim()) return t('teacherQuiz.titleRequired')
    if (!quiz.questions.length) return t('teacherQuiz.needOneQuestion')
    if (quiz.questions.some((q) => !q.text.trim())) return t('teacherQuiz.questionNeedsText')
    const mcqQuestions = quiz.questions.filter((q) => q.type === 'MCQ')
    if (mcqQuestions.some((q) => (q.options || []).length !== 4 || q.options.some((o) => !o.text.trim())))
      return t('teacherQuiz.mcqNeedsFourOptions')
    if (mcqQuestions.some((q) => q.options.filter((o) => o.isCorrect).length !== 1))
      return t('teacherQuiz.mcqNeedsCorrectOption')
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
      showToast(publish ? t('teacherQuiz.publishedToStudents') : t('teacherQuiz.savedAsDraft'))
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
  const updateQuestionType = (index, type) =>
    setQuiz((value) => ({
      ...value,
      questions: value.questions.map((q, i) =>
        i === index
          ? {
              ...q,
              type,
              options:
                type === 'MCQ'
                  ? q.options && q.options.length === 4
                    ? q.options
                    : [0, 1, 2, 3].map((n) => ({
                        id: `opt-${Date.now()}-${n}`,
                        text: '',
                        isCorrect: false,
                      }))
                  : [],
            }
          : q,
      ),
    }))
  const updateOptionText = (qIndex, oIndex, text) =>
    setQuiz((value) => ({
      ...value,
      questions: value.questions.map((q, i) =>
        i === qIndex
          ? { ...q, options: q.options.map((o, j) => (j === oIndex ? { ...o, text } : o)) }
          : q,
      ),
    }))
  const setCorrectOption = (qIndex, oIndex) =>
    setQuiz((value) => ({
      ...value,
      questions: value.questions.map((q, i) =>
        i === qIndex
          ? { ...q, options: q.options.map((o, j) => ({ ...o, isCorrect: j === oIndex })) }
          : q,
      ),
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
        <Loading label={t('teacherQuiz.loadingEditor')} />
      </div>
    )
  return (
    <div className="page page--narrow has-bg-image" style={{ backgroundImage: `url(${teacherQuizBackground})` }}>
      <PageHeader
        eyebrow={id ? t('teacherQuiz.editQuiz') : t('teacherQuiz.newQuiz')}
        title={id ? quiz.title : t('teacherQuiz.createSpeechQuiz')}
        description={t('teacherQuiz.formDescription')}
        back={
          <button className="back-link" onClick={() => navigate('/teacher/quizzes')}>
            {t('teacherQuiz.backToQuizzes')}
          </button>
        }
      />
      {error && <Alert>{error}</Alert>}
      <section className="form-card">
        <h2>{t('teacherQuiz.quizDetails')}</h2>
        <div className="field">
          <label htmlFor="quiz-title">{t('teacherQuiz.titleLabel')}</label>
          <input
            id="quiz-title"
            value={quiz.title}
            onChange={(e) => setQuiz({ ...quiz, title: e.target.value })}
            placeholder={t('teacherQuiz.titlePlaceholder')}
          />
        </div>
        <div className="field">
          <label htmlFor="quiz-description">{t('teacherQuiz.description')}</label>
          <textarea
            id="quiz-description"
            rows="3"
            value={quiz.description}
            onChange={(e) => setQuiz({ ...quiz, description: e.target.value })}
            placeholder={t('teacherQuiz.descriptionPlaceholder')}
          />
        </div>
        <div className="field field--half">
          <label htmlFor="quiz-due">
            {t('teacherQuiz.dueDate')} <span>{t('teacherQuiz.optional')}</span>
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
            <h2>{t('teacherQuiz.questions')}</h2>
            <p>{t('teacherQuiz.questionsDescription')}</p>
          </div>
          <button
            className="button button--secondary button--small"
            onClick={() =>
              setQuiz({
                ...quiz,
                questions: [
                  ...quiz.questions,
                  { id: `q-${Date.now()}`, text: '', type: 'SPOKEN', required: true, options: [] },
                ],
              })
            }
          >
            <Icon name="plus" size={16} /> {t('teacherQuiz.addQuestion')}
          </button>
        </div>
        <div className="question-editor-list">
          {quiz.questions.map((question, index) => (
            <div className="question-editor" key={question.id}>
              <span className="drag-number">{index + 1}</span>
              <div className="field">
                <label htmlFor={`q-text-${question.id}`}>{t('teacherQuiz.questionN', index + 1)}</label>
                <textarea
                  id={`q-text-${question.id}`}
                  lang="si"
                  rows="2"
                  value={question.text}
                  onChange={(e) => updateQuestion(index, e.target.value)}
                  placeholder={t('teacherQuiz.questionPlaceholder')}
                />
              </div>
              <div className="field">
                <label>{t('teacherQuiz.questionType')}</label>
                <div className="filter-tabs" role="tablist">
                  {['SPOKEN', 'MCQ'].map((type) => (
                    <button
                      key={type}
                      type="button"
                      role="tab"
                      aria-selected={(question.type || 'SPOKEN') === type}
                      className={(question.type || 'SPOKEN') === type ? 'active' : ''}
                      onClick={() => updateQuestionType(index, type)}
                    >
                      {type === 'SPOKEN' ? t('teacherQuiz.typeSpoken') : t('teacherQuiz.typeMCQ')}
                    </button>
                  ))}
                </div>
              </div>
              {question.type === 'MCQ' && (
                <div className="field">
                  <label>{t('teacherQuiz.mcqOptions')}</label>
                  {(question.options || []).map((option, oIndex) => (
                    <div className="mcq-option-editor" key={option.id || oIndex}>
                      <input
                        type="radio"
                        name={`correct-${question.id}`}
                        checked={Boolean(option.isCorrect)}
                        onChange={() => setCorrectOption(index, oIndex)}
                        aria-label={t('teacherQuiz.mcqMarkCorrect')}
                      />
                      <input
                        type="text"
                        lang="si"
                        value={option.text}
                        onChange={(e) => updateOptionText(index, oIndex, e.target.value)}
                        placeholder={t('teacherQuiz.mcqOptionPlaceholder', oIndex + 1)}
                      />
                    </div>
                  ))}
                </div>
              )}
              <div className="question-editor__actions">
                <button
                  className="icon-button"
                  aria-label={t('teacherQuiz.moveUp')}
                  disabled={index === 0}
                  onClick={() => move(index, -1)}
                >
                  ↑
                </button>
                <button
                  className="icon-button"
                  aria-label={t('teacherQuiz.moveDown')}
                  disabled={index === quiz.questions.length - 1}
                  onClick={() => move(index, 1)}
                >
                  ↓
                </button>
                <button
                  className="icon-button icon-button--danger"
                  aria-label={t('teacherQuiz.removeQuestion')}
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
          {t('teacherQuiz.saveAsDraft')}
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
          {saving ? t('teacherQuiz.saving') : t('teacherQuiz.publishQuiz')}
        </button>
      </div>
      <ConfirmDialog
        open={removeIndex !== null}
        dangerous
        title={t('teacherQuiz.removeQuestionTitle')}
        message={t('teacherQuiz.removeQuestionMessage')}
        confirmLabel={t('teacherQuiz.removeQuestion')}
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
        title={t('teacherQuiz.publishQuizTitle')}
        message={t('teacherQuiz.publishQuizMessage')}
        confirmLabel={t('teacherQuiz.publishQuiz')}
        onCancel={() => setPublishConfirm(false)}
        onConfirm={() => {
          setPublishConfirm(false)
          save(true)
        }}
      />
    </div>
  )
}
