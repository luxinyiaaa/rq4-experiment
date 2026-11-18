import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const base = process.env.RERUN_SCORING_DIR ?? path.dirname(new URL(import.meta.url).pathname);
const jobs = [
  {
    data: 'alternative_answers_to_review_data.json', output: 'alternative_answers_to_review.xlsx',
    sheet: 'Alternatives to review',
    headers: ['answer_id', 'model', 'question', 'reference_answer', 'decisive_keypoints',
      'answer', 'judge_reason', 'my_decision'],
    widths: [31, 25, 68, 80, 80, 100, 85, 22],
  },
  {
    data: 'relevance_low_scores_data.json', output: 'final/relevance_low_scores.xlsx',
    sheet: 'Low relevance',
    headers: ['experiment', 'answer_id', 'model_or_condition', 'question', 'full_answer_text', 'primary_score',
      'primary_reason', 'secondary_score', 'secondary_reason', 'primary_abstention_label'],
    widths: [15, 31, 25, 68, 100, 18, 85, 18, 85, 27],
  },
];

for (const job of jobs) {
  const rows = JSON.parse(await fs.readFile(path.join(base, job.data), 'utf8'));
  const wb = Workbook.create();
  const sheet = wb.worksheets.add(job.sheet);
  sheet.showGridLines = false;
  sheet.getRangeByIndexes(0, 0, rows.length + 1, job.headers.length).values =
    [job.headers, ...rows.map(row => job.headers.map(key => row[key] ?? ''))];
  const heading = sheet.getRangeByIndexes(0, 0, 1, job.headers.length);
  heading.format.fill = '#183153';
  heading.format.font.color = '#FFFFFF';
  heading.format.font.bold = true;
  heading.format.wrapText = true;
  heading.format.rowHeight = 40;
  sheet.freezePanes.freezeRows(1);
  for (let i = 0; i < job.widths.length; i++)
    sheet.getRangeByIndexes(0, i, 1, 1).format.columnWidth = job.widths[i];
  if (rows.length) {
    const body = sheet.getRangeByIndexes(1, 0, rows.length, job.headers.length);
    body.format.wrapText = true;
    body.format.verticalAlignment = 'top';
    body.format.rowHeight = 240;
  }
  await wb.recalculate();
  const result = await SpreadsheetFile.exportXlsx(wb);
  await result.save(path.join(base, job.output));
  const preview = await wb.render({ sheetName: job.sheet,
    range: `A1:${String.fromCharCode(64 + job.headers.length)}${Math.min(rows.length + 1, 3)}`,
    scale: 0.5, format: 'png' });
  await fs.writeFile(path.join(base, job.output.replace('.xlsx', '_preview.png')),
    new Uint8Array(await preview.arrayBuffer()));
  console.log(`${job.output}: ${rows.length} rows`);
}
