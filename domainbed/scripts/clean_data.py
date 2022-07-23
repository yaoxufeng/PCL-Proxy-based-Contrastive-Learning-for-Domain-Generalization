# coding:utf-8

import os

def clean_domainnet():
	full_path = "/users/smart/Dataset/DG/domain_net"

	with open("./domainbed/misc/domain_net_duplicates.txt", "r")as f:
		for line in f.readlines():
			try:
				os.remove(os.path.join(full_path, line.strip()))
			except OSError:
				pass
			
			
if __name__ == "__main__":
	clean_domainnet()