# startDate = "03/08/2023 4:10 AM"
# endDate = "03/15/2023 4:10 AM"

# sd = startDate.split(" ")
# sd1 = sd[0].split('/')
# sd2 = f'{sd1[2]}-{sd1[0]}-{sd1[1]}'
# print(sd2)


atf = str(({'attendance_id': 1, 'date': '2023-03-15', 'time': '03:26:16', 'student_id': 1903032, 'student_fname': 'Muskan Gupta', 'subject_name': 'Blockchain', 'attendance': 'Present'}, {'attendance_id': 2, 'date': '2023-03-15', 'time': '03:30:48', 'student_id': 1903032, 'student_fname': 'Muskan Gupta', 'subject_name': 'SM', 'attendance': 'Present'}))

print(type(atf))
a = tuple(atf)
print(a, type(a))

# for a in atf:
    # print(type(a))

# print(type(atf[0]))

# a = list(atf)
# print(type(a))