import {feedback} from '../components/common/fields.js';
import {empty, panel, pill} from '../components/common/panels.js';
import {table} from '../components/common/table.js';
import {handler} from '../hooks/action.js';
import {workspace} from '../stores/workspace.js';
import {button, el, h, link, p} from '../utils/dom.js';
import {value} from '../utils/format.js';

function decodeKey(text) {
  try { return JSON.parse(text); } catch { return text.split('|'); }
}

function allocationTable(alloc, predicted, unitRate) {
  const rows = Object.entries(alloc).map(([text, vehicles]) => {
    const [road, intersection, shift] = decodeKey(text);
    return {road, intersection, shift, vehicles,
      rate: unitRate?.[text] ?? null, predicted: predicted?.[text] ?? null};
  }).sort((a, b) => a.road.localeCompare(b.road, 'zh') || a.intersection.localeCompare(b.intersection, 'zh')
    || a.shift.localeCompare(b.shift, 'zh'));
  return table([
    {key:'road', label:'道路'}, {key:'intersection', label:'路口'}, {key:'shift', label:'时段'},
    {key:'vehicles', label:'分配车次'}, {key:'rate', label:'每车次预计率'}, {key:'predicted', label:'预计阻断次数'},
  ], rows, {caption:'可行车次分配', pageSize:8});
}

function dayPanel(day) {
  const status = day.feasible
    ? pill('对得上', 'complete')
    : pill('对不上 · 不出建议', 'failed');
  const signalRows = Object.entries(day.signal_totals || {}).map(([road, total]) => ({road, total}));
  const dutyRows = Object.entries(day.duty_totals || {}).map(([shift, total]) => ({shift, total}));
  const body = [
    el('div', {class:'actions'}, status, day.reason && p(day.reason, 'feedback')),
    el('div', {class:'reconcile-totals'},
      panel('信号台原数（日 × 路段总车次）', table([
        {key:'road', label:'路段'}, {key:'total', label:'信号台总车次'},
      ], signalRows, {caption:'信号台原数'})),
      panel('值班台原数（日 × 全线分时总车次）', table([
        {key:'shift', label:'时段'}, {key:'total', label:'值班台总车次'},
      ], dutyRows, {caption:'值班台原数'}))),
  ];
  if (!day.feasible) {
    return panel(`${day.date} 对账结果`, ...body,
      p('该日期两套总数或漏记区间无可行整数分配，已按口径跳过巡查建议。', 'subtle'));
  }
  const rangeRows = Object.entries(day.ranges).map(([intersection, span]) => ({
    intersection,
    min: span.min, max: span.max,
    robust: day.robust_leaders.includes(intersection),
  })).sort((a, b) => Number(b.robust) - Number(a.robust) || b.max - a.max);
  body.push(panel('各路口预计阻断次数的可行范围', table([
    {key:'intersection', label:'路口'},
    {key:'min', label:'每种分配下的最小值'}, {key:'max', label:'每种分配下的最大值'},
    {label:'每种分配都领先', render: row => row.robust ? pill('是', 'complete') : '否'},
  ], rangeRows, {caption:'可行范围'})));
  const witnesses = Object.values(day.witnesses || {});
  if (witnesses.length) {
    const witnessBlocks = witnesses.map(witness => el('details', {class:'witness'},
      el('summary', {}, `${witness.leader} 并非在每种分配下都领先 ${witness.challenger}（最不利差距 ${value(witness.gap)}）`),
      p('下面是一套满足信号台行和、值班台列和与漏记区间的整数分配，在此分配下上述领先被推翻。', 'subtle'),
      allocationTable(witness.allocation, witness.predicted, day.unit_rate)));
    body.push(panel('可推翻领先的分配', ...witnessBlocks));
  } else {
    body.push(panel('可推翻领先的分配', p('各领先路口在全部可行分配下均不落后，无推翻情形。', 'subtle')));
  }
  const reference = el('details', {class:'witness'},
    el('summary', {}, '查看该日基准可行分配（用于同等工作量情景）'),
    allocationTable(day.reference_allocation, day.reference_predicted, day.unit_rate));
  body.push(panel('基准可行分配', reference));
  return panel(`${day.date} 对账结果`, ...body);
}

export async function reconcilePage(route) {
  const {spec, runs} = workspace;
  const id = route.id || runs[0]?.id;
  if (!id) return panel(spec.pages[4], empty('还没有可查看的运行结果。', '#run', '前往运行'));
  const job = await workspace.getRun(id);
  const message = feedback();
  const picker = el('label', {class:'field run-picker', for:'reconcile-picker'}, el('span', {}, '切换运行记录'),
    el('select', {id:'reconcile-picker', onChange: event => { location.hash = `#reconcile/${event.target.value}`; }},
      (runs.some(item => item.id === job.id) ? runs : [job, ...runs]).map(item =>
        el('option', {value:item.id, selected:item.id === job.id}, item.name))));
  if (job.status !== 'complete' && job.status !== 'failed') {
    return panel(job.name, p('后台正在计算，完成后会自动展示。', 'subtle'),
      link('前往运行状态', `#results/${job.id}`, {class:'quiet'}));
  }
  if (job.status === 'failed' || !job.result) {
    return panel(job.name, p(job.error || '当前运行没有可复核结果。', 'feedback'),
      link('返回调整参数', `#run/${job.dataset_id}`, {class:'quiet'}));
  }
  const result = job.result;
  if (result.mode !== 'reconciliation') {
    return el('div', {}, panel(job.name, picker, message,
      p('该运行基于字段齐全的旧台账，未包含信号台/值班台外部总数或漏记区间，沿用原模型结果。', 'subtle'),
      link('前往路段与路口风险查看原报表', `#results/${job.id}`, {class:'quiet'}),
      el('div', {class:'actions'}, button('下载对账示例 CSV', {class:'quiet', onClick: event =>
        handler(event.currentTarget, message, () =>
          workspace.downloadReconcileTemplate(spec.reconcile_sample_name))(event)}))));
  }
  const days = (result.reconciliation || []).filter(day => day.active);
  const blocked = !result.all_feasible;
  const banner = el('div', {class:`reconcile-banner ${blocked ? 'is-blocked' : 'is-ok'}`, role:'status'},
    h(3, result.headline), p(result.note, 'intro-copy'));
  const download = button('下载结果 CSV', {class:'quiet', 'data-testid':'reconcile-export', onClick: event =>
    handler(event.currentTarget, message, () =>
      workspace.download(`/runs/${job.id}/export`, `${spec.short_name}-${job.id}.csv`))(event)});
  return el('div', {}, panel(job.name, picker, banner, el('div', {class:'actions'},
    link('查看模型与巡查计划', `#results/${job.id}`, {class:'quiet'}), download, message)),
    ...days.map(dayPanel));
}
