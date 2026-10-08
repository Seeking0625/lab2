"""Resource-free startup and transferring a runtime bundle to another checkout."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import app

spec=importlib.util.spec_from_file_location('lab_resources',Path(__file__).resolve().parents[1]/'scripts/resources.py')
resources=importlib.util.module_from_spec(spec);spec.loader.exec_module(resources)

class PortabilityTests(unittest.TestCase):
    def test_empty_checkout_catalogue_and_page(self):
        with tempfile.TemporaryDirectory() as td:
            with patch.object(app,'IMAGE_DIR',td),patch.object(app,'RESULT_DIR',td):
                previous=app.app.config.get('ARCHIVE_IMAGES_ENABLED',True)
                app.app.config['ARCHIVE_IMAGES_ENABLED']=False
                try:
                    client=app.app.test_client()
                    self.assertEqual(client.get('/api/image-catalog').get_json(),[])
                    response=client.get('/')
                    self.assertEqual(response.status_code,200)
                    response.close()
                finally:app.app.config['ARCHIVE_IMAGES_ENABLED']=previous

    def test_bundle_import_checks_integrity_and_preserves_existing(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);content=b'sample resource';name='asset/demo.bin'
            manifest={name:dict(sha256=hashlib.sha256(content).hexdigest(),size=len(content))}
            bundle=root/'bundle.zip'
            with zipfile.ZipFile(bundle,'w') as z:
                z.writestr(name,content);z.writestr('manifest.json',json.dumps(manifest))
            with patch.object(resources,'ROOT',root):
                resources.import_bundle(bundle)
                self.assertEqual((root/name).read_bytes(),content)
                (root/name).write_bytes(b'user resource')
                with self.assertRaises(FileExistsError):resources.import_bundle(bundle)
                self.assertEqual((root/name).read_bytes(),b'user resource')

    def test_lazy_archive_browsing_without_preexisting_images(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'dota/images').mkdir(parents=True);(root/'images').mkdir()
            with zipfile.ZipFile(root/'dota/images/part1.zip','w') as z:z.writestr('images/example.png',b'example image')
            app._archive_images.cache_clear()
            try:
                with patch.object(app.config,'DATA_DIR',root/'dota'),patch.object(app,'IMAGE_DIR',str(root/'images')),patch.object(app,'RESULT_DIR',str(root/'results')):
                    rows=app.app.test_client().get('/api/image-catalog').get_json()
                    self.assertEqual([r['name'] for r in rows],['example.png'])
                    self.assertEqual(list((root/'images').iterdir()),[])
                    path=app._image_path('example.png')
                    self.assertEqual(Path(path).read_bytes(),b'example image')
                    self.assertEqual(len(app.app.test_client().get('/api/image-catalog').get_json()),1)
            finally:app._archive_images.cache_clear()

if __name__=='__main__':unittest.main()
