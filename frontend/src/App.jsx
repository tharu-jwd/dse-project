import { Navigate, Route, Routes } from 'react-router-dom'
import AppShell from './components/AppShell'
import ProtectedRoute from './components/ProtectedRoute'
import CreateTranscriptPage from './pages/CreateTranscriptPage'
import DashboardPage from './pages/DashboardPage'
import LoginPage from './pages/LoginPage'
import NotFoundPage from './pages/NotFoundPage'
import { HelpPage, SettingsPage } from './pages/SettingsHelpPages'
import { SubmissionReviewPage, SubmissionsPage } from './pages/SubmissionPages'
import { QuizAnswerPage, QuizListPage } from './pages/StudentQuizPages'
import { QuizFormPage, TeacherQuizListPage } from './pages/TeacherQuizPages'
import { TranscriptLibraryPage, TranscriptPage } from './pages/TranscriptPages'
import VoiceSampleCollectorPage from './pages/VoiceSampleCollectorPage'

const protectedShell = (
  <ProtectedRoute>
    <AppShell />
  </ProtectedRoute>
)
const teacher = (page) => <ProtectedRoute roles={['TEACHER']}>{page}</ProtectedRoute>
const student = (page) => <ProtectedRoute roles={['STUDENT']}>{page}</ProtectedRoute>

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route element={protectedShell}>
        <Route index element={<Navigate to="/dashboard" replace />} />
        <Route path="dashboard" element={<DashboardPage />} />
        <Route path="lectures/new" element={<CreateTranscriptPage type="LECTURE" />} />
        <Route path="notes/new" element={student(<CreateTranscriptPage type="NOTE" />)} />
        <Route path="transcripts" element={<TranscriptLibraryPage />} />
        <Route path="transcripts/:id" element={<TranscriptPage />} />
        <Route path="quizzes" element={student(<QuizListPage />)} />
        <Route path="quizzes/:id" element={student(<QuizAnswerPage />)} />
        <Route path="teacher/quizzes" element={teacher(<TeacherQuizListPage />)} />
        <Route path="teacher/quizzes/new" element={teacher(<QuizFormPage />)} />
        <Route path="teacher/quizzes/:id/edit" element={teacher(<QuizFormPage />)} />
        <Route path="teacher/submissions" element={teacher(<SubmissionsPage />)} />
        <Route path="teacher/submissions/:id" element={teacher(<SubmissionReviewPage />)} />
        <Route path="settings" element={<SettingsPage />} />
        <Route path="help" element={<HelpPage />} />
        <Route path="dev/voice-samples" element={<VoiceSampleCollectorPage />} />
      </Route>
      <Route path="*" element={<NotFoundPage />} />
    </Routes>
  )
}
