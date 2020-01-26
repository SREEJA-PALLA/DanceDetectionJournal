import cv2
import argparse
import sys
import numpy as np

#####################################################################

keep_processing = True

# parse command line arguments for camera ID or video file

#parser = argparse.ArgumentParser(description='Perform ' + sys.argv[0] + ' example operation on incoming camera/video image')
#parser.add_argument("-c", "--camera_to_use", type=int, help="specify camera to use", default=0)
#parser.add_argument("-r", "--rescale", type=float, help="rescale image by this factor", default=1.0)
#parser.add_argument('video_file', metavar='video_file', type=str, nargs='?', help='specify optional video file')
#args = parser.parse_args()

#####################################################################

# draw optic flow visualization on image using a given step size for
# the line glyphs that show the flow vectors on the image

def draw_flow(img, flow, step=8):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

#####################################################################

# define video capture object
cap=cv2.VideoCapture(0)
fourcc=cv2.VideoWriter_fourcc("X","V","I","D")

video_file=cv2.VideoWriter("gif.mp4",fourcc,20.0,(320,240))
                    
print(cap.isOpened())
while(cap.isOpened()):
    ret, frame =cap.read()
    if ret==True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("frame",frame)
        cv2.imshow("frame",gray)

        if cv2.waitKey(1) &0xFF ==ord('s'):
            break
    else:
        break

# define display window name

windowName = "Dense Optic Flow" # window name

# if command line arguments are provided try to read video_name
# otherwise default to capture from attached H/W camera

if (((args.video_file) and (cap.open(str(args.video_file))))
    or (cap.open(args.camera_to_use))):

    # create window by name (as resizable)

    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

    # if video file successfully open then read an initial frame from video

    if (cap.isOpened):
        ret, frame = cap.read()

        # rescale if specified

        if (args.rescale != 1.0):
            frame = cv2.resize(frame, (0, 0), fx=args.rescale, fy=args.rescale)


    # convert image to grayscale to be previous frame

    prevgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    while (keep_processing):

        # if video file successfully open then read frame from video

        if (cap.isOpened):
            ret, frame = cap.read()

            # when we reach the end of the video (file) exit cleanly

            if (ret == 0):
                keep_processing = False
                continue

            # rescale if specified

            if (args.rescale != 1.0):
                frame = cv2.resize(frame, (0, 0), fx=args.rescale, fy=args.rescale)

        # convert image to grayscale

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # compute dense optic flow using technique of Farneback 2003
        # parameters from example (OpenCV 3.2):
        # https://github.com/opencv/opencv/blob/master/samples/python/opt_flow.py

        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prevgray = gray

        # display image with optic flow overlay

        cv2.imshow(windowName, draw_flow(gray, flow))

        # start the event loop - essential

        # cv2.waitKey() is a keyboard binding function (argument is the time in milliseconds).
        # It waits for specified milliseconds for any keyboard event.
        # If you press any key in that time, the program continues.
        # If 0 is passed, it waits indefinitely for a key stroke.
        # (bitwise and with 0xFF to extract least significant byte of multi-byte response)

        key = cv2.waitKey(40) & 0xFF # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)

        # It can also be set to detect specific key strokes by recording which key is pressed

        # e.g. if user presses "x" then exit  / press "f" for fullscreen display

        if (key == ord('x')):
            keep_processing = False
        elif (key == ord('f')):
            cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # close all windows

    cv2.destroyAllWindows()

else:
    print("No video file specified or camera connected.")
