#!/usr/bin/env python3

import json
import html
import sys


def main(argv: list[str]):
    if len(argv) != 2 or argv[1] == '--help':
        print('Usage: plot_latency.py INPUT.json > OUTPUT.svg')
        print()
    with open(argv[1], 'r', encoding='utf-8') as f:
        data = dict(json.load(f))['inter_thread_latency']
    core_id = dict.keys(data)
    n = len(core_id)
    x_px = [f'{x * 36 + 31.34765625:.8f}' for x in range(n + 1)]
    y_px = [f'{y * 36 + 23.7265625:.7f}' for y in range(n + 1)]
    rtt_sorted = [rtt for host in dict.values(data) for rtt in dict.values(host['rtt_to']) if isinstance(rtt, (int, float)) and rtt > 0]
    rtt_sorted.sort()
    # Each number is 1139/2048em wide, 1466/2048em high
    # Horizontal Margin: 4.65234375px, Vertical margin: 12.2734375px
    print('<svg viewBox="-36 -36 {0} {0}" xmlns="http://www.w3.org/2000/svg">'.format(n * 36 + 72))
    print('  <style>')
    print('    .label {')
    print('      font-family: Arial, sans-serif;')
    print('      font-feature-settings: "tnum";')
    print('    }')
    print('    .value {')
    print('      color: black;')
    print('      font-family: Arial, sans-serif;')
    print('      font-feature-settings: "tnum";')
    print('    }')
    print('  </style>')
    for x, guest in enumerate(core_id):
        print(f'  <text class="label" font-size="16px" text-anchor="end" x="{x_px[x]}" y="-12.2734375">{html.escape(guest)}</text>')
    for y, host in enumerate(core_id):
        print(f'  <text class="label" font-size="16px" text-anchor="end" x="-4.65234375" y="{y_px[y]}">{html.escape(host)}</text>')
        for x, guest in enumerate(core_id):
            if x == y:
                print(f'  <rect shape-rendering="crispEdges" x="{x * 36}" y="{y * 36}" width="36" height="36" fill="{quantile_to_color(0)}"/>')
                continue
            rtt = data[host]['rtt_to'][guest]
            if not (isinstance(rtt, (int, float)) and rtt > 0):
                continue
            rtt_ns = rtt * 1e9
            if rtt_ns < 9.95:
                rtt_str = f'{rtt_ns:.1f}'
            elif rtt_ns < 999.5:
                rtt_str = f'{rtt_ns:.0f}'
            elif rtt_ns < 99500:
                rtt_str = f'{rtt * 1e6:.0f}&#181;'
            elif rtt_ns < 950000:
                rtt_str = f'{rtt * 1e3:.1f}m'.lstrip('0')
            elif rtt_ns < 99500000:
                rtt_str = f'{rtt * 1e3:.0f}m'
            elif rtt_ns < 950000000:
                rtt_str = f'{rtt:.1f}s'.lstrip('0')
            elif rtt_ns < 9500000000:
                rtt_str = f'{rtt:.0f}s'
            else:
                rtt_str = '>9s'
            quantile = (rtt_sorted.index(rtt) + 1) / len(rtt_sorted)
            print('  <g>')
            print(f'    <title>{rtt_ns:.0f}ns</title>')
            print(f'    <rect shape-rendering="crispEdges" x="{x * 36}" y="{y * 36}" width="36" height="36" fill="{quantile_to_color(quantile)}" />')
            print(f'    <text class="value" font-size="16px" text-anchor="end" x="{x_px[x]}" y="{y_px[y]}">{rtt_str}</text>')
            print('  </g>')
        print(f'  <text class="label" font-size="16px" text-anchor="end" x="{x_px[n]}" y="{y_px[y]}">{html.escape(host)}</text>')
    for x, guest in enumerate(core_id):
        print(f'  <text class="label" font-size="16px" text-anchor="end" x="{x_px[x]}" y="{y_px[n]}">{html.escape(guest)}</text>')
    print('</svg>')


def quantile_to_color(quantile: float) -> str:
    lab0 = 0.5726441638642074, -0.18449888213835486, 0.0731887882273366
    lab1 = 0.9377025653501474, -0.043697082175747415, 0.2011401758046535
    lab2 = 0.5870923722305419, 0.20094823015981866, 0.10169546612356578
    if quantile < 0.5:
        s, t = 1 - quantile * 2, quantile * 2
        u, v = lab0, lab1
    else:
        s, t = 2 - quantile * 2, quantile * 2 - 1
        u, v = lab1, lab2
    lab = tuple(s * uu + t * vv for uu, vv in zip(u, v))
    lms = (
        (lab[0] + 0.3963377774 * lab[1] + 0.2158037573 * lab[2])**3,
        (lab[0] - 0.1055613458 * lab[1] - 0.0638541728 * lab[2])**3,
        (lab[0] - 0.0894841775 * lab[1] - 1.2914855480 * lab[2])**3
    )
    rgb = (
        4.0767416621 * lms[0] - 3.3077115913 * lms[1] + 0.2309699292 * lms[2],
		-1.2684380046 * lms[0] + 2.6097574011 * lms[1] - 0.3413193965 * lms[2],
		-0.0041960863 * lms[0] - 0.7034186147 * lms[1] + 1.7076147010 * lms[2],
    )
    rgb = tuple(round(3294.6 * c if c <= 0.0031308 else 269.025 * c**0.4166666666666667 - 14.025) for c in rgb)
    rgb = tuple(min(max(c, 0), 255) for c in rgb)
    return '#{:02x}{:02x}{:02x}'.format(*rgb)


if __name__ == '__main__':
    main(sys.argv)
