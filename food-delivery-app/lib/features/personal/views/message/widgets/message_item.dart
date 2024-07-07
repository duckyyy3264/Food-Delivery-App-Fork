import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:food_delivery_app/common/widgets/cards/circle_icon_card.dart';
import 'package:food_delivery_app/common/widgets/misc/text_with_size.dart';
import 'package:food_delivery_app/utils/constants/icon_strings.dart';
import 'package:food_delivery_app/utils/constants/sizes.dart';

class MessageItem extends StatelessWidget {
  final String title;
  final String description;
  final IconData? icon;
  final String? iconStr;
  final String time;
  final String avatar;
  final Color? iconBgColor;

  MessageItem({
    required this.title,
    required this.description,
    required this.time,
    required this.avatar,
    this.icon,
    this.iconStr,
    this.iconBgColor,
    super.key
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.end,
      children: [
        Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Row(
              children: [
                CircleAvatar(
                  child: Image.asset(
                    avatar
                  )
                ),
                SizedBox(width: TSize.spaceBetweenItemsHorizontal * 2),

                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      title,
                      style: Theme.of(context).textTheme.titleLarge,
                    ),
                    SizedBox(height: TSize.spaceBetweenItemsHorizontal / 2),

                    TextWithSize(
                      aspectScreenWidth: 0.75,
                      text: description,
                      style: Theme.of(context).textTheme.bodyMedium,
                    )
                  ],
                ),
              ],
            ),
          ],
        ),
        Text(
          time,
          style: Theme.of(context).textTheme.bodySmall,
        )
      ],
    );
  }
}
