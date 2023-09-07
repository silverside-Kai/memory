def beginning_of_certain_date(n):
    from datetime import datetime, timedelta

    
    certain_date = datetime.now() - timedelta(days=n)
    beginning = datetime(certain_date.year, certain_date.month, certain_date.day, 0, 0, 0)
    return beginning