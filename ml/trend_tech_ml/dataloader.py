import csv
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict


class TiobeDataLoader:
    """TIOBE index data loader for time series dataset creation"""

    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Directory path containing TIOBE CSV files
        """
        self.data_dir = Path(data_dir)
        self.language_mapping = {
            'python': 'python',
            'java': 'java',
            'c++': 'cpp',
            'c#': 'csharp',
            'c': 'c',
            'javascript': 'javascript',
            'sql': 'sql',
            'r': 'r',
            'visual-basic': 'visual_basic',
            'perl': 'perl'
        }

    def load_csv(self, filename: str) -> dict:
        """
        Load individual CSV file.

        Args:
            filename: CSV filename

        Returns:
            Dictionary with date as key and percent as value
        """
        filepath = self.data_dir / filename
        data = {}

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                date_str = row['date']
                try:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    month_key = date_obj.strftime('%Y-%m')
                    percent = float(row['percent'])
                    data[month_key] = percent
                except (ValueError, KeyError):
                    continue

        return data

    def merge_all_languages(self, languages: list = None) -> dict:
        """
        Merge multiple language data into single dictionary.

        Args:
            languages: List of languages to merge. None for all languages

        Returns:
            Dictionary with date as key and language percentages as nested dict
        """
        if languages is None:
            languages = list(self.language_mapping.keys())

        merged_data = defaultdict(dict)

        for lang_file, lang_col in self.language_mapping.items():
            if lang_file not in languages:
                continue

            csv_filename = f"{lang_file}.csv"

            # Check if CSV file exists
            if not (self.data_dir / csv_filename).exists():
                print(f"Warning: {csv_filename} not found. Skipping...")
                continue

            lang_data = self.load_csv(csv_filename)
            print(f"Loaded {lang_file}: {len(lang_data)} dates")

            for date, percent in lang_data.items():
                merged_data[date][lang_col] = percent

        return merged_data

    def filter_common_period(self, data: dict, required_languages: list, start_date: str = None) -> dict:
        """
        Filter to common period where all required languages have data.

        Args:
            data: Merged data dictionary
            required_languages: List of required language column names
            start_date: Minimum date (YYYY-MM format). If None, use all dates.

        Returns:
            Filtered data dictionary
        """
        filtered_data = {}

        for date, lang_values in data.items():
            # Apply start date filter if specified
            if start_date and date < start_date:
                continue

            # Check if all required languages have values
            if all(lang in lang_values for lang in required_languages):
                filtered_data[date] = lang_values

        return filtered_data

    def save_to_csv(self, data: dict, output_path: str, languages: list):
        """
        Save data to CSV file.

        Args:
            data: Data dictionary to save
            output_path: Output file path
            languages: List of language column names in order
        """
        # Sort dates
        sorted_dates = sorted(data.keys())

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Write header
            header = ['date'] + languages
            writer.writerow(header)

            # Write data rows
            for date in sorted_dates:
                row = [date] + [data[date].get(lang, '') for lang in languages]
                writer.writerow(row)

        print(f"\nData saved to {output_path}")
        print(f"Number of rows: {len(sorted_dates)}")
        if sorted_dates:
            print(f"Date range: {sorted_dates[0]} to {sorted_dates[-1]}")
        print(f"Languages: {', '.join(languages)}")

    def create_integrated_dataset(
        self,
        output_path: str,
        languages: list = None,
        filter_common: bool = True,
        start_from_common: bool = True
    ):
        """
        Create and save integrated dataset.

        Args:
            output_path: Output CSV file path
            languages: List of languages to include (None for all)
            filter_common: Whether to filter to common period
            start_from_common: Start from 2010-09 (when all languages have data)
        """
        print("Loading and merging language data...")
        print("=" * 60)

        if languages is None:
            languages = list(self.language_mapping.keys())

        data = self.merge_all_languages(languages)

        # Get language column names
        lang_cols = [self.language_mapping[lang] for lang in languages
                     if lang in self.language_mapping]

        print(f"\nInitial number of dates: {len(data)}")

        # Determine start date (2010-09 when Visual Basic data starts)
        start_date = "2010-09" if start_from_common else None

        if filter_common:
            print(f"\nFiltering common period...")
            if start_date:
                print(f"Starting from: {start_date} (when all languages have data)")
            print("Removing rows with missing values...")

            data = self.filter_common_period(data, lang_cols, start_date)
            print(f"After filtering: {len(data)} dates")

        print("\n" + "=" * 60)
        print("Saving to CSV...")
        self.save_to_csv(data, output_path, lang_cols)

        return data


if __name__ == "__main__":
    # Usage example

    # Data directory path
    data_dir = "/Users/parkjuyong/Desktop/4-1/CareerRoute/assets/tiobe"

    # Output file path
    output_dir = "/Users/parkjuyong/Desktop/4-1/CareerRoute/ml/trend_tech_ml/data"
    os.makedirs(output_dir, exist_ok=True)  # Create directory if not exists
    output_file = os.path.join(output_dir, "integrated_tiobe_data.csv")

    # Create DataLoader
    loader = TiobeDataLoader(data_dir)

    # Use all available languages
    all_languages = ['python', 'java', 'c++', 'c#', 'c', 'javascript', 'sql', 'r', 'visual-basic', 'perl']

    print("TIOBE Data Integration")
    print("=" * 60)
    print(f"Target languages: {', '.join(all_languages)}")
    print("=" * 60)

    # Create integrated dataset
    # - Starts from 2010-09 (when Visual Basic data begins)
    # - Filters to common period (all languages have data)
    # - Removes rows with missing values
    data = loader.create_integrated_dataset(
        output_path=output_file,
        languages=all_languages,
        filter_common=True,
        start_from_common=True
    )

    print("\n" + "=" * 60)
    print("Dataset creation complete!")
    print("=" * 60)
