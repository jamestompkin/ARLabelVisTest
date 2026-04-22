"""Run the LUT metrics report (hue histograms, gradient stats, alpha-sweep plot).
(Was `Mode.METRICS` in the old main.py.)
"""
from arlabelvis.metrics import run_metrics


def main():
    run_metrics()


if __name__ == "__main__":
    main()
