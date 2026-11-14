
from analyzer import (
    load_loan_data,
    analyze_loan_data,
    prepare_data,
    train_model,
    evaluate_model,
    predict_loan,
    export_results
)


def main():
    print("=" * 50)
    print("       CREDIT RISK PREDICTION SYSTEM")
    print("=" * 50)

    try:
        df = load_loan_data("loan_data.csv")

        analyze_loan_data(df)

        X_train, X_test, y_train, y_test = prepare_data(df)

        model = train_model(X_train, y_train)

        predictions = evaluate_model(model, X_test, y_test)

        predict_loan(
            model,
            income=750000,
            credit_score=740,
            loan_amount=300000,
            employment_years=6,
            existing_loans=1
        )

        export_results(X_test, y_test, predictions)

        print("\nProject completed successfully!")

    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"\nError: {error}")


if __name__ == "__main__":
    main()
