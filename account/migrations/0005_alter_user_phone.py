# Generated by Django 5.0.6 on 2024-11-08 10:59

import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('account', '0004_alter_user_phone_alter_user_tc'),
    ]

    operations = [
        migrations.AlterField(
            model_name='user',
            name='phone',
            field=models.CharField(blank=True, max_length=10, null=True, unique=True, validators=[django.core.validators.RegexValidator('^\\d{10}$')], verbose_name='Phone Number'),
        ),
    ]
