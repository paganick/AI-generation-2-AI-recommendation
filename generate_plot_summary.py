"""
Generate a comprehensive summary visualization of all analysis plots.
Creates both an HTML report and a visual grid showing all available plots.
"""

import os
from pathlib import Path
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
import numpy as np

def find_all_plots():
    """Find all PNG plots in the project."""
    base_dir = Path('.')
    plot_dirs = {
        'Scenario 1': 'scenario1_output/analysis_plots',
        'Scenario 2': 'scenario2_output/analysis_plots',
        'Scenario 3': 'scenario3_output/analysis_plots',
        'Scenario 4': 'scenario4_output/analysis_plots',
        'Multi-Comparison': 'multi_comparison_results',
        'Validation (Self-Pref)': 'validation_results/self_pref_llama_rec/analysis_plots'
    }

    plots_by_category = {}

    for category, dir_path in plot_dirs.items():
        full_path = base_dir / dir_path
        if full_path.exists():
            png_files = sorted(full_path.glob('*.png'))
            if png_files:
                plots_by_category[category] = png_files

    return plots_by_category


def generate_html_report(plots_by_category, output_file='plot_summary_report.html'):
    """Generate an HTML report with all plots."""

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Analysis Plots Summary</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }
            h2 {
                color: #34495e;
                margin-top: 40px;
                background-color: #ecf0f1;
                padding: 10px;
                border-left: 5px solid #3498db;
            }
            .summary-stats {
                background-color: #fff;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .stat-box {
                display: inline-block;
                margin: 10px 20px;
                padding: 15px 25px;
                background-color: #3498db;
                color: white;
                border-radius: 5px;
                font-size: 18px;
                font-weight: bold;
            }
            .plot-container {
                background-color: white;
                padding: 20px;
                margin: 20px 0;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .plot-name {
                font-weight: bold;
                color: #7f8c8d;
                margin-bottom: 10px;
                font-size: 14px;
            }
            img {
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            .toc {
                background-color: #fff;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .toc ul {
                list-style-type: none;
                padding-left: 0;
            }
            .toc li {
                margin: 8px 0;
                padding: 5px;
            }
            .toc a {
                color: #2980b9;
                text-decoration: none;
                font-size: 16px;
            }
            .toc a:hover {
                text-decoration: underline;
            }
            .plot-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }
        </style>
    </head>
    <body>
        <h1>📊 Analysis Plots Summary Report</h1>
    """

    # Count total plots
    total_plots = sum(len(plots) for plots in plots_by_category.values())
    total_categories = len(plots_by_category)

    # Summary statistics
    html_content += f"""
        <div class="summary-stats">
            <h3>Overview</h3>
            <div class="stat-box">{total_categories} Categories</div>
            <div class="stat-box">{total_plots} Total Plots</div>
        </div>
    """

    # Table of contents
    html_content += """
        <div class="toc">
            <h3>Table of Contents</h3>
            <ul>
    """

    for category in plots_by_category.keys():
        category_id = category.replace(' ', '_').replace('(', '').replace(')', '')
        plot_count = len(plots_by_category[category])
        html_content += f'            <li><a href="#{category_id}">{category}</a> ({plot_count} plots)</li>\n'

    html_content += """
            </ul>
        </div>
    """

    # Add each category with plots
    for category, plot_files in plots_by_category.items():
        category_id = category.replace(' ', '_').replace('(', '').replace(')', '')
        html_content += f'\n        <h2 id="{category_id}">{category}</h2>\n'
        html_content += '        <div class="plot-grid">\n'

        for plot_file in plot_files:
            plot_name = plot_file.stem.replace('_', ' ').title()
            rel_path = plot_file.relative_to('.')

            html_content += f"""
            <div class="plot-container">
                <div class="plot-name">{plot_name}</div>
                <img src="{rel_path}" alt="{plot_name}">
                <div style="font-size: 12px; color: #95a5a6; margin-top: 10px;">
                    Path: {rel_path}
                </div>
            </div>
            """

        html_content += '        </div>\n'

    html_content += """
    </body>
    </html>
    """

    # Write HTML file
    with open(output_file, 'w') as f:
        f.write(html_content)

    print(f"✓ HTML report generated: {output_file}")
    return output_file


def create_thumbnail_grid(plots_by_category, output_file='plot_summary_grid.png'):
    """Create a visual grid showing thumbnails of all plots."""

    # Flatten all plots
    all_plots = []
    labels = []

    for category, plot_files in plots_by_category.items():
        for plot_file in plot_files:
            all_plots.append(plot_file)
            plot_name = plot_file.stem.replace('_', ' ').title()
            labels.append(f"{category}\n{plot_name}")

    n_plots = len(all_plots)

    if n_plots == 0:
        print("No plots found!")
        return

    # Calculate grid dimensions
    n_cols = min(3, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))

    # Create figure
    fig = plt.figure(figsize=(20, 6 * n_rows))
    fig.suptitle('Complete Analysis Plots Overview', fontsize=20, fontweight='bold', y=0.995)

    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.3)

    for idx, (plot_file, label) in enumerate(zip(all_plots, labels)):
        row = idx // n_cols
        col = idx % n_cols

        ax = fig.add_subplot(gs[row, col])

        try:
            img = mpimg.imread(plot_file)
            ax.imshow(img)
            ax.set_title(label, fontsize=10, fontweight='bold', pad=10)
            ax.axis('off')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading\n{plot_file.name}',
                   ha='center', va='center', fontsize=10)
            ax.set_title(label, fontsize=10, fontweight='bold', pad=10)
            ax.axis('off')

    # Hide any unused subplots
    for idx in range(n_plots, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        ax.axis('off')

    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Thumbnail grid generated: {output_file}")
    plt.close()


def generate_text_summary(plots_by_category):
    """Generate a text summary of all available plots."""

    print("\n" + "="*70)
    print("PLOT SUMMARY REPORT")
    print("="*70)

    total_plots = sum(len(plots) for plots in plots_by_category.values())

    print(f"\nTotal Categories: {len(plots_by_category)}")
    print(f"Total Plots: {total_plots}")
    print("\n" + "-"*70)

    for category, plot_files in plots_by_category.items():
        print(f"\n📁 {category} ({len(plot_files)} plots)")
        print("-" * 70)
        for plot_file in plot_files:
            plot_name = plot_file.stem.replace('_', ' ').title()
            print(f"  • {plot_name}")
            print(f"    {plot_file.relative_to('.')}")

    print("\n" + "="*70)

    # Plot type summary
    plot_types = {}
    for category, plot_files in plots_by_category.items():
        for plot_file in plot_files:
            plot_name = plot_file.stem
            if plot_name not in plot_types:
                plot_types[plot_name] = []
            plot_types[plot_name].append(category)

    print("\nPLOT TYPES ACROSS SCENARIOS")
    print("="*70)

    for plot_type, categories in sorted(plot_types.items()):
        plot_name = plot_type.replace('_', ' ').title()
        print(f"\n📊 {plot_name}")
        print(f"   Found in: {', '.join(categories)}")

    print("\n" + "="*70)


if __name__ == "__main__":
    print("Generating comprehensive plot summary...")

    # Find all plots
    plots_by_category = find_all_plots()

    if not plots_by_category:
        print("❌ No plots found in expected directories!")
        exit(1)

    # Generate text summary
    generate_text_summary(plots_by_category)

    # Generate HTML report
    html_file = generate_html_report(plots_by_category)

    # Generate thumbnail grid
    create_thumbnail_grid(plots_by_category)

    print("\n" + "="*70)
    print("✓ Summary generation complete!")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  1. plot_summary_report.html - Interactive HTML report")
    print(f"  2. plot_summary_grid.png - Visual thumbnail grid")
    print(f"\nTo view HTML report: open plot_summary_report.html in a browser")
    print("="*70)
