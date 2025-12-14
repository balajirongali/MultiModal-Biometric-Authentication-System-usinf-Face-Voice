import os
import glob
import argparse
import random
from voice_auth import register_user, identify_user, load_db, save_db

# --- Configuration ---
VOICE_DIR = "voice/"
# Number of voices to use for registration for each user.
# The last voice is always reserved for testing.
NUM_VOICES_TO_REGISTER = 5
# Maximum number of test files to use per user.
MAX_TEST_COPIES = 10

# --- Users to register ---
# Add names of folders from VOICE_DIR to this list.
# A user needs at least 2 audio files to be registered and tested.
USERS_TO_REGISTER = [
    "Chetan",
    "Dhanush",
    "Hrisheek",
    "Vivek",
    "kajol",
    "kartheeka",
    "s3",
    "s10",
    "s23",
]

# --- Users to test ---
# Add folders from VOICE_DIR. The key is the user to test, the value is the expected result.
# For registered users, the expected result should be their name.
# For unknown users, the expected result should be "UNKNOWN".
USERS_TO_TEST = {
    "Chetan": "Chetan",  # Expect Chetan's voice to be identified as Chetan
    "Dhanush": "Dhanush",
    "Hrisheek": "Hrisheek",
    "Vivek": "Vivek",
    "kajol": "kajol",
    "kartheeka": "kartheeka",
    "appu": "UNKNOWN",  # Expect appu's voice to be UNKNOWN
    "geetha": "UNKNOWN",
    "sunitha": "UNKNOWN",
    "s10": "s10",
    "s20": "UNKNOWN",
}

# --- Testing parameters ---
IDENTIFICATION_THRESHOLD = 0.8


def clear_database():
    """Wipes the existing user database."""
    print("--- Clearing database ---")
    save_db({})
    print("Database cleared.\n")


def register_users(users_to_register):
    """Registers users specified in USERS_TO_REGISTER."""
    print("--- Starting User Registration ---")
    for user in users_to_register:
        user_dir = os.path.join(VOICE_DIR, user)
        if not os.path.isdir(user_dir):
            print(
                f"⚠️ Warning: Directory not found for user '{user}'. Skipping."
            )
            continue

        audio_files = sorted(glob.glob(os.path.join(user_dir, "*.wav")))
        if len(audio_files) < 2:
            print(
                f"⚠️ Warning: User '{user}' has fewer than 2 audio files. "
                "At least one for registration and one for testing are required. Skipping."
            )
            continue

        # All files except the last are candidates for registration
        registration_candidates = audio_files[:-1]
        # Limit the number of registration files by NUM_VOICES_TO_REGISTER
        files_for_registration = registration_candidates[
            :NUM_VOICES_TO_REGISTER
        ]

        if not files_for_registration:
            print(
                f"⚠️ Warning: No files selected for registration for user '{user}' "
                "(check NUM_VOICES_TO_REGISTER). Skipping."
            )
            continue

        result = register_user(user, files_for_registration)
        print(result)
    print("--- Registration Complete ---\n")


def run_tests(users_to_test, registered_users):
    """Runs identification tests for users specified in USERS_TO_TEST."""
    print("--- Running Identification Tests ---")
    total_tests = 0
    passed_tests = 0

    for user, expected_result in users_to_test.items():
        user_dir = os.path.join(VOICE_DIR, user)
        if not os.path.isdir(user_dir):
            print(
                f"⚠️ Warning: Directory not found for test user '{user}'. Skipping."
            )
            continue

        audio_files = sorted(glob.glob(os.path.join(user_dir, "*.wav")))
        if not audio_files:
            print(
                f"⚠️ Warning: No .wav files found for test user '{user}'. Skipping."
            )
            continue

        test_files = []
        if user in registered_users:
            # For registered users, test files are those not used for registration
            if len(audio_files) < 2:
                print(
                    f"⚠️ Warning: Not enough audio files to test registered user '{user}'. Skipping."
                )
                continue

            registration_candidates = audio_files[:-1]
            files_for_registration_count = min(
                len(registration_candidates), NUM_VOICES_TO_REGISTER
            )
            # The rest of the files are for testing
            all_test_files = audio_files[files_for_registration_count:]

        else:
            # For unknown users, all files are for testing
            all_test_files = audio_files

        if len(all_test_files) > MAX_TEST_COPIES:
            test_files = random.sample(all_test_files, MAX_TEST_COPIES)
        else:
            test_files = all_test_files

        if not test_files:
            print(
                f"⚠️ Warning: No test files found for user '{user}'. Skipping."
            )
            continue

        print(f"--- Testing user: {user} (Expected: {expected_result}) ---")
        for audio_file in test_files:
            total_tests += 1
            result = identify_user(
                audio_file, threshold=IDENTIFICATION_THRESHOLD
            )

            test_passed = False
            if expected_result == "UNKNOWN" and "UNKNOWN" in result:
                test_passed = True
            elif (
                expected_result != "UNKNOWN"
                and f"IDENTIFIED: {expected_result}" in result
            ):
                test_passed = True

            if test_passed:
                passed_tests += 1
                status = "✅ PASS"
            else:
                status = "❌ FAIL"

            print(
                f"[{status}] Testing '{os.path.basename(audio_file)}': Expected='{expected_result}', Got='{result}'"
            )
        print("-" * (len(user) + 26) + "\n")

    print("--- Test Run Summary ---")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print("------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Voice authentication testing script."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Register and test all users in the voice directory.",
    )
    args = parser.parse_args()

    users_to_register = USERS_TO_REGISTER
    users_to_test = USERS_TO_TEST

    if args.all:
        print("--- Running in --all mode: Ignoring configured user lists ---")
        all_users = [
            d
            for d in os.listdir(VOICE_DIR)
            if os.path.isdir(os.path.join(VOICE_DIR, d))
        ]
        users_to_register = all_users
        users_to_test = {user: user for user in all_users}
        print(
            f"Found {len(all_users)} users to register and test: {all_users}\n"
        )

    clear_database()
    register_users(users_to_register)
    run_tests(users_to_test, users_to_register)
