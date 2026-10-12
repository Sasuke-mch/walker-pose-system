"""Compact presentation for the current Canvas viewer, without changing data."""
import html
import re


def minimal_viewer_text(page, title="人体与助步器"):
    """Preserve the embedded payload and runtime IDs while shortening the UI."""
    payload = re.search(
        r'<script id="data" type="application/json">.*?</script>', page, re.S)
    if payload:
        return (minimal_viewer_text(page[:payload.start()], title) + payload[0]
                + minimal_viewer_text(page[payload.end():], title))
    escaped_title = html.escape(title)
    page = re.sub(r'<title>.*?</title>', lambda _: '<title>' + escaped_title + '</title>', page)
    page = re.sub(r'<h1>.*?</h1>(?:<p>.*?</p>)?',
                  lambda _: '<h1>' + escaped_title + '</h1>', page)
    sidebar = re.search(r'<aside class="side">.*?</aside>', page, re.S)
    if sidebar:
        checks = re.search(r'<div class="checks">.*?</div>', sidebar[0], re.S)
        if checks is None:
            raise ValueError('viewer display controls missing')
        controls = checks[0]
        for old, new in (('扶手范围', '扶手'), ('手脚采样点', '手脚点'),
                         ('三角化点', '观测点'), ('相机轨迹', '相机')):
            controls = controls.replace(old, new)
        smoothing = ''
        if 'id="smoothModel"' in sidebar[0]:
            smoothing = ('<section><label><input id="smoothModel" '
                         'type="checkbox" checked> 平滑</label></section>')
        compact = ('<aside class="side"><section><dl class="metrics">'
                   '<div><dt>地面</dt><dd id="pose">—</dd></div></dl></section>'
                   '<section><h2>显示</h2>' + controls + '</section>' + smoothing
                   + '<section><details><summary>详情</summary>'
                   '<pre id="qa" style="white-space:pre-wrap;font-size:12px"></pre>'
                   '</details></section><div hidden><span id="wrist"></span>'
                   '<span id="e3"></span><span id="hr"></span><span id="fr"></span>'
                   '<pre id="weights"></pre></div></aside>')
        page = page[:sidebar.start()] + compact + page[sidebar.end():]
    page = page.replace('重置视角', '复位')
    page = page.replace("protocol_support_restored:'协议支撑闭合'",
                        "protocol_support_restored:'闭合'")
    page = page.replace("protocol_support_window_reestimated:'过渡窗口重估'",
                        "protocol_support_window_reestimated:'重估'")
    page = page.replace("accepted_rotation_held:'更新（旋转保持）'",
                        "accepted_rotation_held:'更新'")
    return page
