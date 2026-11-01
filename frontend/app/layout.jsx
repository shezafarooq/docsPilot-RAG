import './globals.css';

export const metadata = {
  title: 'DocsPilot — Ask your docs',
  description: 'Answers from Docker and FastAPI documentation, with sources you can inspect.',
};

export default function RootLayout({ children }) {
  return <html lang="en"><body>{children}</body></html>;
}
