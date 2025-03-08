app_landmarkOnly.py:
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload


# plux :
#  def onRawFrame(self, nSeq, data):
#         if not stop_flag.is_set():
#             if nSeq % 50 == 0:
#                 row=[nSeq]+list(data)
#                 self.save_to_excel(row)
#                 data_queue.put(data)
#         else:
#             print("escaped0")
#             return True

will be looping and occupy the thread until return True.




2025/2/14
这个是回埼玉之后的版本
