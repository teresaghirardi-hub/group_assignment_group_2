## AI Usage Declaration

This project made targeted use of AI assistance (Claude by Anthropic and ChatGPT by OpenAI)
in the following areas only:

---

### What AI was used for

**UI/UX and visual design**
AI tools were used to explore colour schemes, suggest CSS layout approaches, and iterate
on the visual theme of the Streamlit interface — including the dark indigo/violet palette,
sidebar styling, calendar date picker overrides, and team card layout. No application logic
was generated through this process.

**Troubleshooting and debugging**
AI assistance was used to identify and fix runtime errors — including duplicate Plotly
element IDs in Streamlit, sklearn version compatibility issues, CSS specificity conflicts,
and edge cases in the backtesting simulation such as empty trade logs and confidence
threshold filtering. In all cases the underlying logic and decisions were our own —
AI helped identify where the problem was, not how the system should work.

**Code cleanup**
AI was used to improve code readability — variable naming, inline comments, and
formatting — without changing any underlying logic.

**Documentation and presentation structure**
AI was used to help structure the README.md and homepage content in app.py — taking
advantage of its ability to think in clear, logical steps to organise information in
a way that is easy to follow. The technical content — feature descriptions, model
explanations, and pipeline steps — is entirely our own and reflects our understanding
of the project.

**Project naming and branding**
AI was consulted for app name suggestions and logo prompt generation for use with
external image generation tools.

---

### What AI was NOT used for

All machine learning concepts, pipeline design, and implementation were completed
independently, based on material from our Machine Learning course. This includes:

- Feature engineering and technical indicator selection
- Model selection methodology (AUC-ROC vs F1 Macro, TimeSeriesSplit rationale)
- Binary and multi-class classification pipeline design
- Backtesting strategy logic (Simple and Advanced modes)
- The SimFin API wrapper (pysimfin.py)
- ETL pipeline architecture and target label construction
- All model evaluation and comparison

The ML notebooks were built using code patterns and concepts taught directly in the
course. AI was not used to write, explain, or generate any part of the machine learning
or data processing codebase.

---

### Summary

AI tools acted as a helpful assistant for presentation, aesthetics, troubleshooting,
and documentation structure — not as a replacement for understanding. Every decision
about the system's architecture, methodology, and implementation was made by the team.