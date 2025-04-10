# MotionMind

## Enable Webcam for WSL

```shell
gsudo usbipd bind -b 6-2
gsudo usbipd attach -b 6-2 -a -w
```