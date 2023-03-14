startDate = "03/08/2023 4:10 AM"
endDate = "03/15/2023 4:10 AM"

sd = startDate.split(" ")
sd1 = sd[0].split('/')
sd2 = f'{sd1[2]}-{sd1[0]}-{sd1[1]}'
print(sd2)

