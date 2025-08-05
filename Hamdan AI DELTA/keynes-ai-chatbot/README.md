# Keynes AI Chatbot

A minimal, elegant React + Tailwind CSS frontend for a financial AI chatbot named "Keynes".

## Features

- **Welcome Page**: Clean, centered full-screen layout with professional styling
- **React Router**: Navigation between pages
- **Tailwind CSS**: Modern, responsive design inspired by OpenAI's ChatGPT interface
- **Professional UI**: Light gray background with modern sans-serif fonts

## Getting Started

### Prerequisites

- Node.js (version 14 or higher)
- npm or yarn

### Installation

1. Clone the repository or navigate to the project directory
2. Install dependencies:
   ```bash
   npm install
   ```

### Running the Application

Start the development server:
```bash
npm start
```

The application will open in your browser at `http://localhost:3000`.

### Building for Production

To create a production build:
```bash
npm run build
```

## Project Structure

```
src/
├── components/
│   ├── WelcomePage.js    # Welcome page with centered layout
│   └── ChatPage.js       # Chat interface (placeholder)
├── App.js                # Main app with routing
├── index.js              # Entry point
└── index.css             # Tailwind CSS imports
```

## Pages

### Welcome Page (`/`)
- Large heading: "Welcome to Keynes"
- Subheading: "Your personal value investing AI assistant"
- Primary button: "Enter Keynes" (navigates to `/chat`)
- Clean, professional styling with light gray background

### Chat Page (`/chat`)
- Placeholder for chat interface
- Navigation back to welcome page
- Ready for chat functionality implementation

## Technologies Used

- **React 18**: Modern React with hooks
- **React Router**: Client-side routing
- **Tailwind CSS**: Utility-first CSS framework
- **@tailwindcss/forms**: Enhanced form styling

## Design Philosophy

The design follows modern UI/UX principles:
- Clean, minimal aesthetic
- Professional color scheme (grays and blues)
- Responsive design
- Smooth transitions and hover effects
- Inspired by OpenAI's ChatGPT interface

## Future Enhancements

- Implement actual chat functionality
- Add authentication
- Integrate with AI backend
- Add more interactive features
- Implement dark mode toggle
