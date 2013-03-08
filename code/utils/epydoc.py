"""
Post-processing of epydoc generated documentation.
"""

import os
import sys
from glob import glob

# adds MathJax capabilities to documentation
injection = """
  <script type="text/x-mathjax-config">
    MathJax.Hub.Config({
      tex2jax: {
        inlineMath: [['$','$']],
        displayMath: [['$$','$$']],
        skipTags: ["script","noscript","style","textarea","code"],
        processEnvironments: true
      }
    });
  </script>
  <script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
  """

def main(argv):
	with open('epydoc.config') as handle:
		for line in handle:
			if line.startswith('target:'):
				target = line.split(':', 1)[1].strip()
				break
		else:
			target = 'doc'

	if not os.path.exists(target):
		os.system('epydoc --config epydoc.config')

	for filename in glob(os.path.join(target, '*.html')):
		html = ''
		with open(filename) as handle:
			for line in handle:
				if 'MathJax.Hub.Config' in line:
					# looks like the documentation has already been processed
					return 0
				if '</head>' in line:
					html += injection
				html += line
		with open(filename, 'w') as handle:	
			handle.writelines(html)
	return 0

if __name__ == '__main__':
	sys.exit(main(sys.argv))
