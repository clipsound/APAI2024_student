import csv
import random
import string
import pandas as pd
import matplotlib.pyplot as plt


def generate_random_value(options_list):
    # Generate a random value from a list of options
    return random.choice(options_list)

def generate_random_numeric_value(min_value, max_value):
    # Generate a random numeric value within the specified range
    return random.randint(min_value, max_value)


def generate_uid():
    # Generate a random UID of 8 characters (combination of letters and numbers)
    characters = string.ascii_letters + string.digits
    return ''.join(random.choices(characters, k=16))

def generate_csv_file(file_name, num_rows, columns):
    with open(file_name, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write header with field names
        csv_writer.writerow(columns.keys())
        # Generate rows with random values
        for _ in range(num_rows):
            row = []
            for field, options in columns.items():
                if isinstance(options, list):
                    # Textual field, choose from a list of options
                    row.append(generate_random_value(options))
                elif isinstance(options, tuple):
                    # Numeric field, generate a random value within the specified range
                    row.append(generate_random_numeric_value(options[0], options[1]))
            row.append(generate_uid())
            csv_writer.writerow(row)

def generate_dataframe(num_rows, columns):
    data = []
    for _ in range(num_rows):
        row = {}
        for field, options in columns.items():
            if isinstance(options, list):
                row[field] = generate_random_value(options)
            elif isinstance(options, tuple):
                row[field] = generate_random_numeric_value(options[0], options[1])
        row['UID'] = generate_uid()
        data.append(row)
    return pd.DataFrame(data)


def main():
    csv_file_name = 'generated_data.csv'
    num_rows = 1000000
    columns = {
        'Name': ['John', 'Paul', 'Mary', 'Lucy'],
        'Surname': ['White', 'Black', 'McDonald', 'Potter'],
        'Age': (1, 100),
        'Bank_Account': (1000, 5000),
        'State': ["California",
                  "New York",
                  "Texas",
                  "Florida",
                  "Illinois",
                  "Pennsylvania",
                  "Ohio",
                  "Georgia",
                  "Michigan",
                  "North Carolina"]
    }


    df = generate_dataframe(num_rows, columns)
    df.to_csv('generated_data_from_dataframe.csv', index=False)
    df.to_pickle('generated_data_from_dataframe.pkl')
    print(df.head())

    plt.hist(df['Age'], bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title('Distribution of Age')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

