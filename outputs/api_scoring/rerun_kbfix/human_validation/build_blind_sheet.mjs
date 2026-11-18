/** Build the blind review workbook. Model identities and judge labels stay in separate files. */
import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const folder = path.dirname(new URL(import.meta.url).pathname);
const { items } = JSON.parse(await fs.readFile(path.join(folder, 'blind_sample_data.json'), 'utf8'));
if (items.length !== 40 || new Set(items.map(x => x.blind_item_id)).size !== 40)
  throw new Error('Expected 40 distinct blind items');

const wb = Workbook.create();
const sheet = wb.worksheets.add('Blind review');
sheet.showGridLines = false;
const headers = ['Blind item ID', 'Question', 'Reference answer', 'Keypoints', 'Full answer text',
  ...Array.from({ length: 8 }, (_, i) => `Keypoint ${i + 1} label`),
  'Abstention', 'Relevance (1–5)', 'Correctness (formula)'];
const letter = n => {
  let s = '';
  for (let x = n + 1; x; x = Math.floor((x - 1) / 26)) s = String.fromCharCode(65 + (x - 1) % 26) + s;
  return s;
};
const table = [headers, ...items.map(x => [
  x.blind_item_id, x.question, x.reference_answer,
  x.keypoints.map(k => `${k.number}. [${k.decisive ? 'DECISIVE' : 'NOT DECISIVE'}] ${k.text}`).join('\n'),
  x.full_answer_text, ...Array(11).fill('')
])];
sheet.getRangeByIndexes(0, 0, 41, 16).values = table;
const head = sheet.getRange('A1:P1');
head.format.fill = '#183153';
head.format.font = { name: 'Aptos', size: 11, bold: true, color: '#FFFFFF' };
head.format.wrapText = true;
head.format.rowHeight = 42;
sheet.getRange('A2:P41').format.font = { name: 'Aptos', size: 11 };
sheet.getRange('A2:P41').format.verticalAlignment = 'top';
sheet.getRange('A2:P41').format.rowHeight = 125;
sheet.getRange('B2:E41').format.wrapText = true;
sheet.getRange('A:A').format.columnWidth = 17;
sheet.getRange('B:B').format.columnWidth = 62;
sheet.getRange('C:C').format.columnWidth = 58;
sheet.getRange('D:D').format.columnWidth = 70;
sheet.getRange('E:E').format.columnWidth = 82;
sheet.getRange('F:M').format.columnWidth = 20;
sheet.getRange('N:O').format.columnWidth = 21;
sheet.getRange('P:P').format.columnWidth = 22;
sheet.getRange('F2:O41').format.fill = '#EAF3FC';
sheet.getRange('P2:P41').format.fill = '#E8F3EA';
sheet.freezePanes.freezeRows(1);

for (let i = 0; i < items.length; i++) {
  const row = i + 2;
  const keypoints = items[i].keypoints;
  for (let j = 0; j < keypoints.length; j++)
    sheet.getRange(`${letter(5 + j)}${row}`).dataValidation = {
      rule: { type: 'list', values: ['Entailment', 'Neutral', 'Contradiction'] }
    };
  sheet.getRange(`N${row}`).dataValidation = {
    rule: { type: 'list', values: ['answers', 'partly_answers', 'abstains'] }
  };
  sheet.getRange(`O${row}`).dataValidation = {
    rule: { type: 'list', values: ['1', '2', '3', '4', '5'] }
  };
  const first = `F${row}`;
  const last = `${letter(4 + keypoints.length)}${row}`;
  const decisive = keypoints.map((k, j) => k.decisive ? `${letter(5 + j)}${row}="Entailment"` : null)
    .filter(Boolean);
  const formula = `=IF(COUNTBLANK(${first}:${last})>0,"",IF(AND(${decisive.join(',')},COUNTIF(${first}:${last},"Contradiction")=0),"Correct","Incorrect"))`;
  sheet.getRange(`P${row}`).formulas = [[formula]];
}

wb.recalculate();
const check = await wb.inspect({ kind: 'table', range: 'Blind review!A1:P3',
  include: 'values,formulas', tableMaxRows: 3, tableMaxCols: 16, maxChars: 3000 });
console.log(check.ndjson);
const result = await SpreadsheetFile.exportXlsx(wb);
await result.save(path.join(folder, 'human_validation_blind.xlsx'));
console.log('Saved 40-item blind workbook');
