export const DEMO_USERS = {
  student: {
    id: 'user-student',
    name: 'Nethmi Perera',
    nameSi: 'නෙත්මි පෙරේරා',
    email: 'student@sinhaspeech.lk',
    password: 'demo123',
    role: 'STUDENT',
  },
  teacher: {
    id: 'user-teacher',
    name: 'Dr. Kasun Silva',
    nameSi: 'ආචාර්ය කසුන් සිල්වා',
    email: 'teacher@sinhaspeech.lk',
    password: 'demo123',
    role: 'TEACHER',
  },
}

const segments = [
  {
    id: 'segment-1', startTime: 0, endTime: 6.4, confidence: 0.94,
    text: 'අද අපි පරිගණක විද්‍යාවේ මූලික සංකල්ප ගැන සාකච්ඡා කරමු.',
    words: [
      { text: 'අද', confidence: 0.98 }, { text: 'අපි', confidence: 0.97 },
      { text: 'පරිගණක', confidence: 0.93 }, { text: 'විද්‍යාවේ', confidence: 0.88 },
      { text: 'මූලික', confidence: 0.95 }, { text: 'සංකල්ප', confidence: 0.72 },
      { text: 'ගැන', confidence: 0.96 }, { text: 'සාකච්ඡා', confidence: 0.91 },
      { text: 'කරමු.', confidence: 0.97 },
    ],
  },
  {
    id: 'segment-2', startTime: 6.4, endTime: 13.2, confidence: 0.78,
    text: 'ඇල්ගොරිතමයක් යනු ගැටලුවක් විසඳීමට යොදා ගන්නා පියවර මාලාවකි.',
    words: [
      { text: 'ඇල්ගොරිතමයක්', confidence: 0.69 }, { text: 'යනු', confidence: 0.96 },
      { text: 'ගැටලුවක්', confidence: 0.74 }, { text: 'විසඳීමට', confidence: 0.82 },
      { text: 'යොදා', confidence: 0.92 }, { text: 'ගන්නා', confidence: 0.89 },
      { text: 'පියවර', confidence: 0.93 }, { text: 'මාලාවකි.', confidence: 0.86 },
    ],
  },
  {
    id: 'segment-3', startTime: 13.2, endTime: 20.8, confidence: 0.92,
    text: 'නිවැරදි ප්‍රතිඵලයක් ලබා ගැනීමට එම පියවර අනුපිළිවෙළින් ක්‍රියාත්මක කළ යුතුයි.',
    words: [
      { text: 'නිවැරදි', confidence: 0.95 }, { text: 'ප්‍රතිඵලයක්', confidence: 0.91 },
      { text: 'ලබා', confidence: 0.97 }, { text: 'ගැනීමට', confidence: 0.94 },
      { text: 'එම', confidence: 0.96 }, { text: 'පියවර', confidence: 0.91 },
      { text: 'අනුපිළිවෙළින්', confidence: 0.77 }, { text: 'ක්‍රියාත්මක', confidence: 0.88 },
      { text: 'කළ', confidence: 0.97 }, { text: 'යුතුයි.', confidence: 0.96 },
    ],
  },
]

export const initialTranscripts = [
  {
    id: 'transcript-lecture-1', ownerId: 'user-student', title: 'Introduction to Algorithms',
    type: 'LECTURE', status: 'DRAFT', date: '2026-07-28T09:30:00Z', mediaUrl: '',
    segments: structuredClone(segments),
  },
  {
    id: 'transcript-note-1', ownerId: 'user-student', title: 'Database revision notes',
    type: 'NOTE', status: 'FINALIZED', date: '2026-07-25T14:10:00Z', mediaUrl: '',
    segments: [{ ...structuredClone(segments[0]), id: 'note-segment-1', text: 'දත්ත සමුදායක් තොරතුරු සංවිධානාත්මකව ගබඩා කිරීමට භාවිතා කරයි.' }],
  },
  {
    id: 'transcript-answer-1', ownerId: 'user-student', title: 'Data Structures — Question 1',
    type: 'QUIZ_ANSWER', status: 'FINALIZED', date: '2026-07-23T11:00:00Z', mediaUrl: '',
    segments: [{ ...structuredClone(segments[1]), id: 'answer-segment-1', text: 'ස්ටැක් එකක් අවසානයේ ඇතුළත් කළ අගය මුලින් පිටතට ලබා දෙයි.' }],
  },
  {
    id: 'transcript-lecture-2', ownerId: 'user-teacher', title: 'Database Systems — Week 04',
    type: 'LECTURE', status: 'FINALIZED', date: '2026-07-29T08:00:00Z', mediaUrl: '',
    segments: structuredClone(segments),
  },
]

export const initialQuizzes = [
  {
    id: 'quiz-1', title: 'Data Structures — Week 03',
    description: 'Answer each question clearly in Sinhala using a short spoken response.',
    status: 'PUBLISHED', dueDate: '2026-08-15', submissionStatus: 'NOT_STARTED',
    questions: [
      { id: 'q-1', text: 'ස්ටැක් දත්ත ව්‍යුහය කෙටියෙන් විස්තර කරන්න.', required: true },
      { id: 'q-2', text: 'Queue එකක් සහ Stack එකක් අතර වෙනස කුමක්ද?', required: true },
      { id: 'q-3', text: 'Linked List භාවිතා කිරීමේ එක් වාසියක් සඳහන් කරන්න.', required: true },
    ],
  },
  {
    id: 'quiz-2', title: 'Database Fundamentals', description: 'Draft quiz for next week.',
    status: 'DRAFT', dueDate: '', submissionStatus: 'NOT_STARTED',
    questions: [{ id: 'q-4', text: 'Primary key යනු කුමක්ද?', required: true }],
  },
]

export const initialSubmissions = [
  {
    id: 'submission-1', quizId: 'quiz-1', quizTitle: 'Data Structures — Week 03',
    studentId: 'user-student', studentName: 'Nethmi Perera', submittedAt: '2026-07-30T10:15:00Z',
    status: 'SUBMITTED', mark: '', feedback: '',
    answers: [
      { questionId: 'q-1', question: 'ස්ටැක් දත්ත ව්‍යුහය කෙටියෙන් විස්තර කරන්න.', transcript: 'ස්ටැක් යනු අවසානයේ ඇතුළත් කළ අගය මුලින් පිටතට දෙන දත්ත ව්‍යුහයකි.' },
      { questionId: 'q-2', question: 'Queue එකක් සහ Stack එකක් අතර වෙනස කුමක්ද?', transcript: 'Queue එක පළමුව ඇතුළත් කළ අගය මුලින් ලබා දෙන අතර Stack එක එහි ප්‍රතිවිරුද්ධ ලෙස ක්‍රියා කරයි.' },
      { questionId: 'q-3', question: 'Linked List භාවිතා කිරීමේ එක් වාසියක් සඳහන් කරන්න.', transcript: 'අවශ්‍ය පරිදි මතකය ගතිකව වෙන් කළ හැකිය.' },
    ],
  },
]

export const sampleSegments = () => structuredClone(segments)
