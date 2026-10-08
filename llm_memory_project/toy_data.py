"""
toy_data.py  -- the 10 CLEAN memory units (LoCoMo-style, hand written).
Two simulated people (Priya, Rohan), 5 sessions.
"""
from schema import MemoryUnit


def load_clean_units():
    return [
        MemoryUnit("m01", "Priya moved to Bangalore in March 2023 to join Infosys as a data analyst.", 1, "2023-03-10"),
        MemoryUnit("m02", "Priya's brother Rohan lives in Pune, automotive startup.", 1, "2023-03-12"),
        MemoryUnit("m03", "Rohan is into trekking, did the Kedarkantha trek last December.", 2, "2023-05-02"),
        MemoryUnit("m04", "Rohan visited Priya in Bangalore, June 2023.", 2, "2023-06-20"),
        MemoryUnit("m05", "Priya mentioned she is learning guitar on weekends.", 3, "2023-08-14"),
        MemoryUnit("m06", "Priya got promoted to senior analyst at Infosys in October 2023.", 3, "2023-10-05"),
        MemoryUnit("m07", "Rohan's startup, VoltRide, raised a seed round.", 4, "2023-11-01"),
        MemoryUnit("m08", "Priya adopted a cat named Momo in November 2023.", 4, "2023-11-18"),
        MemoryUnit("m09", "Priya and Rohan planned a family trip to Goa, New Year 2024.", 5, "2023-12-20"),
        MemoryUnit("m10", "Priya switched jobs and joined Flipkart in January 2024.", 5, "2024-01-15"),
    ]


# Which clean unit holds the CORRECT answer for each test query
# (this is the ground truth Module 4 scores against).
TEST_QUERIES = {
    "Where does Priya work now?": ["m10"],
    "What did Rohan do in 2023?": ["m03", "m04", "m07"],
}
