# Generated by Django 3.2.7 on 2024-08-04 09:27

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('deliverer', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='otherinfo',
            old_name='other_info',
            new_name='deliverer',
        ),
    ]