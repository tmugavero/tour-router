export const metadata = {
  title: 'Wilder AI — Tour Recommendation Engine',
  description: 'Cross-artist ML tour market recommendations',
}
export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body style={{ margin: 0, padding: 0 }}>{children}</body>
    </html>
  )
}
