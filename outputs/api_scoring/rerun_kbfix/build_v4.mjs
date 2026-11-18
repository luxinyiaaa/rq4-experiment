/** Build the rerun workbook using the v3 sheet layout and columns. */
import fs from 'node:fs/promises';
import path from 'node:path';
import { FileBlob, Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const base = process.env.RERUN_SCORING_DIR ?? path.dirname(new URL(import.meta.url).pathname);
const data = JSON.parse(await fs.readFile(path.join(base, 'v4_workbook_data.json'), 'utf8'));
const { headers, rows } = data;
const templatePath = path.resolve(path.dirname(new URL(import.meta.url).pathname),
  '../full300/final/all_answers_and_scores_v3.xlsx');
const template = await SpreadsheetFile.importXlsx(await FileBlob.load(templatePath));
const templateHeaders = template.worksheets.getItem('All answers and scores')
  .getRangeByIndexes(0, 0, 1, 24).values[0];
if (JSON.stringify(templateHeaders) !== JSON.stringify(headers))
  throw new Error('v4 headers differ from v3');
const readmeRows = template.worksheets.getItem('README')
  .getRangeByIndexes(0, 0, 40, 2).values.map(row => [row[0] ?? '', row[1] ?? '']);
while (readmeRows.length && !readmeRows.at(-1)[0] && !readmeRows.at(-1)[1]) readmeRows.pop();
if (readmeRows[0][0] !== 'How to read the keypoint labels' || readmeRows[9][0] !== 'Column')
  throw new Error('Unexpected v3 README layout');
readmeRows.splice(8, 0, ['Collection start (UTC)', data.collection_start_utc],
  ['Collection end (UTC)', data.collection_end_utc]);
if (headers.length !== 24 || rows.length !== 1200 ||
    new Set(rows.map(row => row[0] + '__' + row[1])).size !== 1200 ||
    new Set(rows.map(row => row[0])).size !== 300)
  throw new Error('Unexpected workbook row count or duplicate question–model pair');

const wb = Workbook.create();
const sheet = wb.worksheets.add('All answers and scores');
sheet.showGridLines = false;
sheet.getRangeByIndexes(0, 0, 1201, 24).values = [headers, ...rows];
const heading = sheet.getRangeByIndexes(0, 0, 1, 24);
heading.format.fill = '#183153';
heading.format.font.color = '#FFFFFF';
heading.format.font.bold = true;
heading.format.wrapText = true;
heading.format.rowHeight = 42;
sheet.freezePanes.freezeRows(1);
for (let i = 0; i < headers.length; i++) {
  const name = headers[i];
  const width = name === 'full_llm_answer' ? 90 :
    ['question','reference_answer','keypoints','primary_correctness_reason',
     'primary_keypoint_labels','secondary_keypoint_labels'].includes(name) ? 75 :
    name === 'source_url' ? 55 : name === 'question_id' ? 31 : 23;
  sheet.getRangeByIndexes(0, i, 1, 1).format.columnWidth = width;
  if (name.endsWith('_completeness') || name.endsWith('_citation_support') ||
      name.endsWith('_citation_completeness') || name.endsWith('_relevance'))
    sheet.getRangeByIndexes(1, i, 1200, 1).format.numberFormat = '0.00';
  if (['question','reference_answer','keypoints','full_llm_answer','primary_correctness_reason',
       'primary_keypoint_labels','secondary_keypoint_labels'].includes(name))
    sheet.getRangeByIndexes(1, i, 1200, 1).format.wrapText = true;
}
const body = sheet.getRangeByIndexes(1, 0, 1200, 24);
body.format.rowHeight = 60;
body.format.verticalAlignment = 'top';

const readme = wb.worksheets.add('README');
readme.showGridLines = false;
readme.getRangeByIndexes(0, 0, readmeRows.length, 2).values = readmeRows;
readme.getRangeByIndexes(0, 0, 1, 2).format.font.bold = true;
readme.getRangeByIndexes(1, 0, 9, 1).format.font.bold = true;
readme.getRangeByIndexes(0, 1, readmeRows.length, 1).format.wrapText = true;
readme.getRangeByIndexes(0, 0, 1, 2).format.fill = '#DDE8F3';
readme.getRangeByIndexes(0, 0, 1, 2).format.rowHeight = 36;
readme.getRangeByIndexes(1, 0, 9, 2).format.rowHeight = 48;
readme.getRangeByIndexes(11, 0, 1, 2).format.fill = '#183153';
readme.getRangeByIndexes(11, 0, 1, 2).format.font.color = '#FFFFFF';
readme.getRangeByIndexes(11, 0, 1, 2).format.font.bold = true;
readme.getRangeByIndexes(0, 0, 1, 1).format.columnWidth = 34;
readme.getRangeByIndexes(0, 1, 1, 1).format.columnWidth = 118;
readme.getRangeByIndexes(12, 0, readmeRows.length - 12, 2).format.rowHeight = 42;
readme.freezePanes.freezeRows(1);

const result = await SpreadsheetFile.exportXlsx(wb);
await result.save(path.join(base, 'final/all_answers_and_scores_v4.xlsx'));
console.log(JSON.stringify({rows: rows.length, sheets: ['All answers and scores','README']}));
