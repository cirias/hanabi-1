import argparse
import json

def main(video_file):
    """
    When you evaluate a policy on a gym environment (see run_ai.py and
    run_self.py), gym produces a directory with a bunch of video files showing
    some sample executions. The video files are named something like
    openaigym.video.0.17502.video010000.json and look something like this:

        {
            "env": {},
            "title": "gym VideoRecorder episode",
            "duration": 8.5,
            "height": 11,
            "command": "-",
            "width": 301,
            "version": 1,
            "stdout": [[0.5, "frame 0"], [0.5, "frame 1"]]
        }

    This script will print out the video frames. You can then pipe them into
    something like less -r to view them.
    """
    with open(video_file, "r") as f:
        j = json.load(f)
        for stdout in j["stdout"]:
            # Every frame begins with some control characters that look like:
            #
            #   \u001b[2J\u001b[1;1Hdeck:
            #
            # We strip off these control characters when printing.
            print(stdout[1][10:])

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_file",
        type=str, help="JSON video file produced by OpenAI Gym.")
    return parser

if __name__ == "__main__":
    main(get_parser().parse_args().video_file)
