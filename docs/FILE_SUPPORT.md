# File handling

## Inputs

- PDF: text extraction with pypdf.
- PPTX: slide text and table extraction with python-pptx.
- DOCX: paragraph and table extraction with python-docx.
- XLSX: worksheet values via openpyxl.
- TXT, Markdown, CSV: UTF-8 text extraction.
- PNG, JPG, JPEG, WEBP: raw image input passed to configured vision-capable providers.

Uploads are limited to 10 MB per file and 20 MB per request. Original attachments are linked to a conversation interaction. Removing a conversation removes those attachments.

## Outputs

- PDF: ReportLab document with headings and paragraph styling.
- DOCX: editable Word document.
- PPTX: editable PowerPoint with title and content slides.
- TXT / Markdown: text export.
- Image: optional Gemini image-generation API, using a separately configured image model.

AI text output is transformed into a downloadable file. It is not the same as a provider directly returning a native Office document. Image generation is disabled until an image-generation model is configured.
