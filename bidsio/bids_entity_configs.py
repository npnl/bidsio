from collections import OrderedDict
# defines the entity name -> path entity name (e.g. subject -> sub) via short_name, and vice versa via long_name
# As of BIDS 1.7, these are in order.
entity_short_name = OrderedDict([
    ('subject', 'sub'),
    ('session', 'ses'),
    ('sample', 'sample'),
    ('task', 'task'),
    ('acquisition', 'acq'),
    ('ceagent', 'ce'),
    ('tracer', 'trc'),
    ('stain', 'stain'),
    ('reconstruction', 'rec'),
    ('direction', 'dir'),
    ('run', 'run'),
    ('modality', 'mod'),
    ('echo', 'echo'),
    ('flip', 'flip'),
    ('inversion', 'inv'),
    ('mt', 'mt'),
    ('part', 'part'),
    ('proc', 'proc'),
    ('hemi', 'hemi'),
    ('space', 'space'),
    ('split', 'split'),
    ('recording', 'recording'),
    ('chunk', 'chunk'),
    ('resolution', 'res'),
    ('den', 'den'),
    ('label', 'label'),
    ('desc', 'desc')
])

entity_long_name = OrderedDict()
for k, v in entity_short_name.items():
    entity_long_name[v] = k